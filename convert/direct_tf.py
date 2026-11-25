"""Direct, FP32 TensorFlow port of model.py. No ONNX/grid_sample bridge.

NHWC externally; NATTEN windows keep explicit head layout internally.
Only emit a published .tflite after TF and Lite pass numerical gates on TWO
PyTorch fixtures. This checks conversion parity, NOT original CUDA model parity
or restoration quality. See docs/implementation-status.md.

Run: venv/bin/python convert/direct_tf.py --prepared convert/prepared
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorflow as tf


class Port(tf.Module):
    def __init__(self,root):
        super().__init__()
        self.weights=dict(np.load(root/'weights.npz'))
        self.meta=json.loads((root/'metadata.json').read_text())
        self.cfg=self.meta['config']
        self.constants={}
        self.chunk=4
        # Share static gather indices across blocks. Embedding a fresh H*W
        # index tensor AND expanded learned bias for every tap/block exceeds
        # protobuf's 2 GiB FunctionDef limit even for this ~24M-param network.
        self.windows={}
        self.references={}
        size=self.meta['size']
        for level,kernel in enumerate(self.cfg['kernel_size']):
            h=w=size//(2**level)
            yy,xx=np.meshgrid(np.linspace(.5,h-.5,h,dtype=np.float32),np.linspace(.5,w-.5,w,dtype=np.float32),indexing='ij')
            self.references[str(h)]=tf.constant(np.stack([yy/(h-1)*2-1,xx/(w-1)*2-1],axis=-1))[None,...]
            for k,d in [(kernel,1),(kernel,2),(kernel+2,1)]:
                sy,py=self.window_axis(h,k,d); sx,px=self.window_axis(w,k,d)
                taps=[(a,z) for a in range(k) for z in range(k)]
                indices=[]; bias_indices=[]; counts=[]
                for start in range(0,len(taps),self.chunk):
                    group=taps[start:start+self.chunk]; counts.append(len(group))
                    indices.append(tf.constant(np.stack([((sy[:,None]+a*d)*w+sx[None,:]+z*d).reshape(-1) for a,z in group],axis=-1).astype(np.int32)))
                    bias_indices.append(tf.constant(np.stack([((py[:,None]+a)*(2*k-1)+px[None,:]+z).reshape(-1) for a,z in group],axis=-1).astype(np.int32)))
                self.windows[f'{h}:{w}:{k}:{d}']={'indices':indices,'bias_indices':bias_indices,'counts':counts}

    def weight(self,name):
        if name not in self.constants:
            self.constants[name]=tf.constant(self.weights[name],tf.float32)
        return self.constants[name]

    def conv(self,x,name,depthwise=False):
        w=self.weights[name+'.weight']
        if depthwise:
            filt=tf.transpose(self.weight(name+'.weight'),[2,3,0,1])
            y=tf.nn.depthwise_conv2d(x,filt,[1,1,1,1],padding='SAME')
        else:
            filt=tf.transpose(self.weight(name+'.weight'),[2,3,1,0])
            y=tf.nn.conv2d(x,filt,[1,1,1,1],padding='SAME')
        if name+'.bias' in self.weights: y=y+self.weight(name+'.bias')
        return y

    def norm(self,x,name):
        mean=tf.reduce_mean(x,axis=-1,keepdims=True)
        var=tf.reduce_mean(tf.square(x-mean),axis=-1,keepdims=True)
        return (x-mean)*tf.math.rsqrt(var+1e-5)*self.weight(name+'.weight')+self.weight(name+'.bias')

    @staticmethod
    def grid_sample(x,grid):
        # PyTorch grid_sample(... align_corners=True, padding_mode='zeros').
        b,h,w,c=x.shape
        gx=(grid[...,1]+1)*(w-1)/2; gy=(grid[...,0]+1)*(h-1)/2
        x0=tf.floor(gx); y0=tf.floor(gy); dx=gx-x0; dy=gy-y0
        batches=tf.broadcast_to(tf.range(b,dtype=tf.int32)[:,None,None],(b,h,w))
        def at(xx,yy):
            valid=(xx>=0)&(xx<w)&(yy>=0)&(yy<h)
            ids=tf.stack([batches,tf.cast(tf.clip_by_value(yy,0,h-1),tf.int32),tf.cast(tf.clip_by_value(xx,0,w-1),tf.int32)],axis=-1)
            return tf.gather_nd(x,ids)*tf.cast(valid[...,None],tf.float32)
        return ((1-dx)*(1-dy))[...,None]*at(x0,y0)+(dx*(1-dy))[...,None]*at(x0+1,y0)+((1-dx)*dy)[...,None]*at(x0,y0+1)+(dx*dy)[...,None]*at(x0+1,y0+1)

    @staticmethod
    def window_axis(n,k,d):
        i=np.arange(n); residue=i%d; count=(n-1-residue)//d+1
        if count.min()<k: raise ValueError('NATTEN spatial resolution too small')
        start=np.minimum(np.maximum(i//d-k//2,0),count-k)*d+residue
        pb=k-1-(i-start)//d
        return start,pb

    def attention_window(self,q,k,v,size,dilation,bias):
        b,g,h,w,c=q.shape
        window=self.windows[f'{h}:{w}:{size}:{dilation}']
        flatq=tf.reshape(q,[b,g,h*w,c])
        flatk=tf.reshape(k,[b,g,h*w,c]); flatv=tf.reshape(v,[b,g,h*w,c])
        # For finite activations this is exactly zero. The runtime dependency
        # prevents constant folding a small RPB table into ~1 GiB of repeated
        # spatial maps across all blocks. Nonfinite activations are invalid.
        runtime_zero=flatq[0,0,0,0]*tf.constant(0.,tf.float32)
        table=tf.reshape(self.weight(bias)+runtime_zero,[g,(2*size-1)**2])
        all_scores=[]
        for idx,pb in zip(window['indices'],window['bias_indices']):
            keys=tf.gather(flatk,idx,axis=2)  # B,G,HW,taps,C (rank 5)
            bb=tf.gather(table,pb,axis=1)     # G,HW,taps
            all_scores.append(tf.reduce_sum(tf.expand_dims(flatq,3)*keys,axis=-1)+bb[None,...])
        probs=tf.nn.softmax(tf.concat(all_scores,axis=-1),axis=-1)
        output=tf.zeros_like(flatq); start=0
        for idx,count in zip(window['indices'],window['counts']):
            probability=tf.expand_dims(probs[:,:,:,start:start+count],axis=-1)
            output=output+tf.reduce_sum(probability*tf.gather(flatv,idx,axis=2),axis=3)
            start+=count
        return tf.reshape(output,[b,g,h,w,c])

    def attention(self,x,name):
        b,h,w,channels=x.shape
        heads=self.weights[name+'.rpb_h1'].shape[0]*3
        c=channels//heads
        q=self.conv(x,name+'.proj_q')
        grouped=tf.reshape(tf.transpose(tf.reshape(q,[b,h,w,heads,c]),[0,3,1,2,4]),[b*heads,h,w,c])
        offset=self.conv(grouped,name+'.conv_offset.0',True)
        offset=self.norm(offset,name+'.conv_offset.1')
        offset=tf.nn.gelu(offset,approximate=False)
        offset=self.conv(offset,name+'.conv_offset.3')
        offset=tf.math.tanh(offset)*tf.constant([1/(h-1),1/(w-1)],tf.float32)
        pos=offset+self.references[str(h)]
        grouped_x=tf.reshape(tf.transpose(tf.reshape(x,[b,h,w,heads,c]),[0,3,1,2,4]),[b*heads,h,w,c])
        sampled=self.grid_sample(grouped_x,pos)
        sampled=tf.reshape(tf.transpose(tf.reshape(sampled,[b,heads,h,w,c]),[0,2,3,1,4]),[b,h,w,channels])
        def split(t): return tf.split(tf.transpose(tf.reshape(t,[b,h,w,heads,c]),[0,3,1,2,4]),3,axis=1)
        qs=split(q*(c**-.5)); ks=split(self.conv(sampled,name+'.proj_k')); vs=split(self.conv(sampled,name+'.proj_v'))
        kernel=self.weights[name+'.conv_offset.0.weight'].shape[2]
        outputs=[self.attention_window(qs[i],ks[i],vs[i],k,d,name+f'.rpb_h{i+1}') for i,(k,d) in enumerate([(kernel,1),(kernel,2),(kernel+2,1)])]
        out=tf.reshape(tf.transpose(tf.concat(outputs,axis=1),[0,2,3,1,4]),[b,h,w,channels])
        return self.conv(out+self.conv(q,name+'.rpe_table',True),name+'.proj_out')

    def stage(self,x,name):
        count=sum(k.startswith(name+'.') and k.endswith('.norm1.body.weight') for k in self.weights)
        for i in range(count):
            p=f'{name}.{i}'
            x=x+self.attention(self.norm(x,p+'.norm1.body'),p+'.attn')
            f=self.conv(self.norm(x,p+'.norm2.body'),p+'.ffn.project_in')
            a,b=tf.split(self.conv(f,p+'.ffn.dwconv',True),2,axis=-1)
            x=x+self.conv(tf.nn.gelu(a,approximate=False)*b,p+'.ffn.project_out')
        return x

    def down(self,x,name):
        x=self.conv(x,name+'.body.0'); b,h,w,c=x.shape
        # PyTorch PixelUnshuffle groups offsets INSIDE each original channel.
        return tf.reshape(tf.transpose(tf.reshape(x,[b,h//2,2,w//2,2,c]),[0,1,3,5,2,4]),[b,h//2,w//2,c*4])

    def up(self,x,name):
        x=self.conv(x,name+'.body.0'); b,h,w,c=x.shape
        return tf.reshape(tf.transpose(tf.reshape(x,[b,h,w,c//4,2,2]),[0,1,4,2,5,3]),[b,h*2,w*2,c//4])

    def __call__(self,x):
        l1=self.stage(self.conv(x,'patch_embed.proj'),'encoder_level1')
        l2=self.stage(self.down(l1,'down1_2'),'encoder_level2')
        l3=self.stage(self.down(l2,'down2_3'),'encoder_level3')
        latent=self.stage(self.down(l3,'down3_4'),'latent')
        d3=self.stage(self.conv(tf.concat([self.up(latent,'up4_3'),l3],-1),'reduce_chan_level3'),'decoder_level3')
        d2=self.stage(self.conv(tf.concat([self.up(d3,'up3_2'),l2],-1),'reduce_chan_level2'),'decoder_level2')
        d1=self.stage(tf.concat([self.up(d2,'up2_1'),l1],-1),'decoder_level1')
        out=self.conv(self.stage(d1,'refinement'),'output')+x
        return out+x if self.meta['global_residual'] else out


def compare(ref,got,stage,tolerance):
    if got.shape!=ref.shape or not np.isfinite(got).all(): raise RuntimeError(f'{stage}: invalid output')
    maximum=float(np.max(np.abs(ref-got))); rmse=float(np.sqrt(np.mean((ref-got)**2)))
    print(stage,'max_abs',maximum,'rmse',rmse,flush=True)
    if maximum>tolerance: raise RuntimeError(f'{stage}: numerical parity failed ({maximum} > {tolerance})')
    return dict(max_abs=maximum,rmse=rmse)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--prepared',default='convert/prepared'); ap.add_argument('--tolerance',type=float,default=1e-3); ap.add_argument('--tf-only',action='store_true')
    args=ap.parse_args(); root=Path(args.prepared); port=Port(root); size=port.meta['size']
    inputs=[np.load(root/f'input_{i}.npy') for i in range(2)]; refs=[np.load(root/f'reference_{i}.npy') for i in range(2)]
    tf_results=[]
    for i,(x,ref) in enumerate(zip(inputs,refs)):
        tf_results.append(compare(ref,port(tf.constant(x)).numpy(),f'TF fixture {i}',args.tolerance))
    if args.tf_only: return
    print('Tracing TensorFlow graph (no ONNX)…',flush=True)
    # Build constants outside tf.function; symbolic tensors must not leak out
    # of its FuncGraph via the cache on the module.
    for key in port.weights: port.weight(key)
    fn=tf.function(port.__call__,input_signature=[tf.TensorSpec([1,size,size,3],tf.float32,name='image')])
    concrete=fn.get_concrete_function()
    print('Traced nodes:',len(concrete.graph.get_operations()),flush=True)
    converter=tf.lite.TFLiteConverter.from_concrete_functions([concrete],port)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations=[]
    data=converter.convert()
    print('Converted bytes:',len(data),flush=True)
    interpreter=tf.lite.Interpreter(model_content=data,num_threads=4)
    interpreter.allocate_tensors(); inp=interpreter.get_input_details()[0]; out=interpreter.get_output_details()[0]
    if inp['dtype']!=np.float32 or out['dtype']!=np.float32: raise RuntimeError('Expected FP32 IO')
    results=[]
    for i,(x,ref) in enumerate(zip(inputs,refs)):
        interpreter.set_tensor(inp['index'],x); interpreter.invoke()
        results.append(compare(ref,interpreter.get_tensor(out['index']),f'TFLite fixture {i}',args.tolerance))
    path=root/f'dswinir_{size}_fp32.tflite'; path.write_bytes(data)
    (root/'conversion-report.json').write_text(json.dumps(dict(checkpoint=port.meta,tf=tf_results,tflite=results,tolerance=args.tolerance,
        tflite_sha256=hashlib.sha256(data).hexdigest(),original_cuda_parity='NOT_VERIFIED',quality_on_test_set='NOT_VERIFIED'),indent=2)+'\n')
    print('Saved',path,flush=True)

if __name__=='__main__': main()
