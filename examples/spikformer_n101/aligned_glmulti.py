"""GLMulti with wrapper-style spike prediction/reset, and exact discrete VJP."""
import torch
from torch import nn
from torch.nn import functional as F

FORWARD = r'''
extern "C" __global__ void aligned_forward(const float* x,const float* v0,
 const float* a,const float* w,float* s,float* p,float* hist,int T,int N,float tau,float th) {
 int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N)return;
 float v=v0[n];
 for(int t=0;t<T;++t) {
  float d=(x[t*N+n]-v)/tau, pred=v+d, spike=pred>=th?1.f:0.f;
  float h=0;for(int j=0;j<t;++j)h+=w[t*T+j]*hist[j*N+n];
  v=a[t]*(d-spike*th/tau)-h;
  s[t*N+n]=spike;p[t*N+n]=pred;hist[t*N+n]=v;
 }
}
'''
BACKWARD = r'''
extern "C" __global__ void aligned_backward(const float* g,const float* a,const float* w,
 const float* p,float* gx,float* gv,float* ga,int T,int N,float tau,float th,float sg) {
 int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N)return;
 float adj[64];for(int t=0;t<T;++t)adj[t]=0;
 for(int t=T-1;t>=0;--t) {
  float z=1.5707963267948966f*sg*(p[t*N+n]-th);
  float q=g[t*N+n]*sg/(2.f*(1.f+z*z)),r=adj[t];
  ga[t*N+n]=r;gx[t*N+n]=(q+r*a[t])/tau;
  float prev=q*(1.f-1.f/tau)-r*a[t]/tau;
  if(t)adj[t-1]+=prev;else gv[n]=prev;
  for(int j=0;j<t;++j)adj[j]-=w[t*T+j]*r;
 }
}
extern "C" __global__ void aligned_scale(const float* ga,const float* hist,const float* w,
 const float* a,float* out,int T,int N,int chunks) {
 int t=blockIdx.y,n=blockIdx.x*blockDim.x+threadIdx.x;float v=0;
 if(n<N){float h=0;for(int j=0;j<t;++j)h+=w[t*T+j]*hist[j*N+n];
  v=ga[t*N+n]*(hist[t*N+n]+h)/a[t];}
 __shared__ float sums[256];sums[threadIdx.x]=v;__syncthreads();
 for(int d=128;d;d>>=1){if(threadIdx.x<d)sums[threadIdx.x]+=sums[threadIdx.x+d];__syncthreads();}
 if(threadIdx.x==0)out[t*chunks+blockIdx.x]=sums[0];
}
'''
_extension = None
_kernels = None
BACKEND = 'native'
CALLS = {'forward': 0, 'backward': 0}
HISTORY_RHO = 1.

def extension():
    global _extension
    if _extension is None:
        from torch.utils.cpp_extension import load_inline
        source = '#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <c10/cuda/CUDAGuard.h>\n#include <c10/cuda/CUDAException.h>\n' + FORWARD + BACKWARD + r'''
std::vector<torch::Tensor> run_forward(torch::Tensor x,torch::Tensor v,torch::Tensor a,torch::Tensor w,double tau,double th){
 c10::cuda::CUDAGuard guard(x.device());int T=x.size(0),N=x.size(1);
 auto s=torch::empty_like(x),p=torch::empty_like(x),h=torch::empty_like(x);
 aligned_forward<<<(N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),v.data_ptr<float>(),a.data_ptr<float>(),w.data_ptr<float>(),s.data_ptr<float>(),p.data_ptr<float>(),h.data_ptr<float>(),T,N,tau,th);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {s,p,h};}
std::vector<torch::Tensor> run_backward(torch::Tensor g,torch::Tensor a,torch::Tensor w,torch::Tensor p,torch::Tensor h,double tau,double th,double sg){
 c10::cuda::CUDAGuard guard(g.device());int T=g.size(0),N=g.size(1),chunks=(N+255)/256;
 auto gx=torch::empty_like(g),ga=torch::empty_like(g),gv=torch::empty({N},g.options()),out=torch::empty({T,chunks},g.options());
 aligned_backward<<<chunks,256,0,at::cuda::getCurrentCUDAStream()>>>(g.data_ptr<float>(),a.data_ptr<float>(),w.data_ptr<float>(),p.data_ptr<float>(),gx.data_ptr<float>(),gv.data_ptr<float>(),ga.data_ptr<float>(),T,N,tau,th,sg);
 aligned_scale<<<dim3(chunks,T),256,0,at::cuda::getCurrentCUDAStream()>>>(ga.data_ptr<float>(),h.data_ptr<float>(),w.data_ptr<float>(),a.data_ptr<float>(),out.data_ptr<float>(),T,N,chunks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {gx,gv,ga,out};}
'''
        declarations = 'std::vector<torch::Tensor> run_forward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,double);\nstd::vector<torch::Tensor> run_backward(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,double,double,double);'
        _extension = load_inline(name='glmulti_wrapper_aligned_v1',cpp_sources=declarations,
            cuda_sources=source,functions=['run_forward','run_backward'],
            extra_cuda_cflags=['-O3','--fmad=false'],extra_cflags=['-O2'])
    return _extension

def cupy_launch(name, arrays, constants, grid):
    global _kernels
    import cupy as cp
    if _kernels is None or _kernels.get('rho')!=HISTORY_RHO:
        code=BACKWARD.replace('adj[j]-=w[t*T+j]*r;',f'adj[j]-={HISTORY_RHO:.8f}f*w[t*T+j]*r;')
        _kernels = {n: cp.RawKernel(FORWARD+code,n,options=('--fmad=false',))
                    for n in ['aligned_forward','aligned_backward','aligned_scale']}
        _kernels['rho']=HISTORY_RHO
    with cp.cuda.Device(arrays[0].device.index), cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream):
        views=[cp.from_dlpack(t.detach()) for t in arrays]
        _kernels[name](grid,(256,),tuple(views)+tuple(constants))

class AlignedGLMulti(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,v,a,w,tau,th,sg,batch,full=False):
        if not full or not 0<x.shape[0]<=64:
            raise ValueError('Requires full history and 1<=T<=64')
        CALLS['forward']+=1
        ctx.dtype=x.dtype;ctx.backend=BACKEND;ctx.constants=(tau,th,sg)
        x,v,a,w=[z.float().contiguous() for z in (x,v,a,w)]
        if ctx.backend=='native':s,p,h=extension().run_forward(x,v,a,w,tau,th)
        else:
            import cupy as cp
            T,N=x.shape;s,p,h=[torch.empty_like(x) for _ in range(3)]
            cupy_launch('aligned_forward',[x,v,a,w,s,p,h],
                [cp.int32(T),cp.int32(N),cp.float32(tau),cp.float32(th)],((N+255)//256,))
        ctx.save_for_backward(a,w,p,h)
        return s.to(ctx.dtype)

    @staticmethod
    def backward(ctx,g):
        CALLS['backward']+=1
        a,w,p,h=ctx.saved_tensors;g=g.float().contiguous();T,N=g.shape
        if ctx.backend=='native':gx,gv,ga,out=extension().run_backward(g,a,w,p,h,*ctx.constants)
        else:
            import cupy as cp
            tau,th,sg=ctx.constants;chunks=(N+255)//256
            gx,ga=torch.empty_like(g),torch.empty_like(g)
            gv=torch.empty(N,device=g.device);out=torch.empty((T,chunks),device=g.device)
            cupy_launch('aligned_backward',[g,a,w,p,gx,gv,ga],
                [cp.int32(T),cp.int32(N),cp.float32(tau),cp.float32(th),cp.float32(sg)],(chunks,))
            cupy_launch('aligned_scale',[ga,h,w,a,out],
                [cp.int32(T),cp.int32(N),cp.int32(chunks)],(chunks,T))
        gs=out.sum(1) if ctx.needs_input_grad[2] else None
        gw=-(ga@h.t())*torch.tril(torch.ones_like(w),diagonal=-1) if ctx.needs_input_grad[3] else None
        return gx.to(ctx.dtype),gv,gs,gw,None,None,None,None,None

def reference(x,v,a,w,tau=2.,th=1.,sg=5.):
    from spikeDE_dis import surrogate
    hist=[];spikes=[]
    for t in range(x.shape[0]):
        d=(x[t]-v)/tau;s=surrogate.arctan_surrogate(v+d-th,sg)
        f=d-s.detach()*th/tau
        v=a[t]*f-(sum(w[t,j]*(hist[j].detach()+HISTORY_RHO*(hist[j]-hist[j].detach())) for j in range(t)) if t else 0.)
        hist.append(v);spikes.append(s)
    return torch.stack(spikes)

def install(backend):
    global BACKEND
    BACKEND=backend
    if backend=='native':extension()
    from spikeDE_dis import fde_cupy_kernels_all as kernels
    kernels.LIFFDEGLMultiFunction=AlignedGLMulti

def make_model(backend,beta,learn,masses,batch=32,T=16,fixed_gain=False):
    from model import DirectFDESpikformer
    from spikeDE_dis.neuron import LIFNeuronFDE
    install(backend)
    raw=nn.Parameter(torch.log(torch.expm1(torch.tensor(masses)))) if learn else None
    initial_c0=sum(m*.1**(-b) for m,b in zip(masses,beta))
    class Neuron(LIFNeuronFDE):
        def get_multi_coefficient(self):
            m=super().get_multi_coefficient()
            return m*(initial_c0/(m*self.multi_beta.new_tensor([.1**(-b) for b in beta])).sum()) if fixed_gain else m
    def factory(name):
        return Neuron(tau=2.,threshold=1.,surrogate_grad_scale=5.,surrogate_function='arctan',
            beta=beta[0],T=T,method='glmulti',step_size=.1,batchsize=batch,name=name,
            backend='cupy',step_mode='m',multi_beta=beta,multi_coefficient=masses,
            learn_coefficient=learn,coefficient_constraint='softplus',shared_multi_coefficient=raw,
            coefficient_update_interval=1,glmulti_history_grad='full')
    net=DirectFDESpikformer(neuron_factory=factory,in_channels=2,num_classes=101,
                          embed_dims=256,depths=2,time_steps=T)
    # The supplementary Block accepts DropPath but does not execute it.
    for block in net.blocks:block.drop_path=nn.Identity()
    net.ablation_implementation=backend
    return net
