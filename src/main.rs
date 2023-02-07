// The data struct is an output from each layer, and is used as an input in the subsequent layer.
// On a few occasions the data is used in later layers and needs to be kept.
// mutable access to a data struct is only required during creation
// all other accesses can use read only values.

use std::vec;

 fn main(){
    
#[derive(Debug)]
struct Data{                                     
    height:u16,
    width:u16,
    depth:u16,
    count:usize,
    data:Vec<f32>,
} 
impl Data {
    fn new (size:[u16;3])->Data{
        Data {
            width:size[0],   //row length
            height:size[1],  // num rows
            depth:size[2],  // num layers
            count:(size[0]*size[1]*size[2]).into(),
            data: vec![0.0; (size[0]*size[1]*size[2]).into()],
        }    
    }
    fn get (&self, x:u16, y:u16, z:u16)-> &f32{             
        let index:usize = (z * (self.height*self.width) + y*self.width +x).into();
        &self.data[index]
    }
    fn set (&mut self, x:u16, y:u16, z:u16, exp:f32) {             
        let index:usize = (z * (self.height*self.width) + y*self.width +x).into();
        self.data[index] = exp;
    }             
} 

#[derive(Debug)]
struct Weights{
    ksize_h:u16,
    ksize_w:u16,
    ch_in:u16,
    ch_out:u16,
    count:usize,
    data:Vec<f32>,
}

impl Weights {
    fn new (&self, size:[u16;4], path:&str)->Weights{
        Weights {
            ksize_h:size[0],
            ksize_w:size[1],
            ch_in:size[2],
            ch_out:size[3],
            count:(self.ksize_h * self.ksize_w * self.ch_in * self.ch_out).into(),
            data: vec![0.0; self.count]
        }            
    } 
    fn get (&self, x:u16, y:u16, z:u16, a:u16)-> &f32{        
        let index:usize = (a*(self.ksize_h*self.ksize_w*self.ch_in) + z*(self.ksize_h * self.ksize_w) + y*(self.ksize_w) +x).into();
        &self.data[index]
    }          
}

fn conv(in_channels:u16, out_channels:u16, ksize:u16, s:u16, p:u16, d:u16, weights:&Weights, data:&Data) -> Data{
      
    let out_height = ((data.height+2*p-d*(ksize-1) )-1/s)+1;
    let out_width = ((data.width+2*p-d*(ksize-1) )-1/s)+1;
    let out_size:[u16;3] = [out_channels,  out_height,  out_width];
    let num_strides_high = data.height/s;
    let num_strides_wide = data.width/s;
    let mut multiply_buffer= Data::new(out_size);
    
    for a in 0..out_channels{
       //let mut total = 0;
       //let mut count = 0;
        for b in 0..in_channels{
            for i in 0..num_strides_high{
                for j in 0..num_strides_wide{
                    for l in 0..ksize{
                        for k in 0..ksize{
                            multiply_buffer.set(a, i, j, (data.get(b, (i*s) + (k*d)-1, (j*s)+(l*d)-1) * weights.get(a, b, l, k)));
                        }
                    }
                }
            }
        }
    }
    return multiply_buffer;
}

fn maxpool(data:Data, ksize:u16, s:u16 , p:u16) { 
    let d = 0; // no dilation in maxpool function.
    let num_strides_high = data.height/s;
    let num_strides_wide = data.width/s;   
    let in_channels = data.depth;
    
    for b in 0..in_channels {
        for i in 0..num_strides_high {
            for j in 0..num_strides_wide {
                let pool_max=0.0;
                for l in 0..ksize {
                    for k in 0..ksize {
                        if data.get(b,(i*s) + (k*d), (j*s)+(l*d) )> pool_max {
                            pool_max = data.get(b,(i*s) + (k*d), (j*s)+(l*d));
                        }
                    }
                }
                for l in 0..ksize {
                    for k in 0..ksize {
                        data.set(b,(i*s) + (k*d), (j*s)+(l*d),pool_max); 
                    }
                }    
            }
        }
    }
}
 
 fn concat(data1: Data, data2: Data)->Data{
    
     //as with all other parameters, data len is precalculated.
     let width = data1.width + data2.width;   //row length
     let height = data1.height + data2.height;  // num rows
     let depth = data1.depth + data2.depth;  // num layers
     let out_size = [width,height,depth];
     let mut concat_buffer= Data::new(out_size);

     for a in 0..width{
        for b in 0..height{
            for c in 0..depth{
                
            }
        }
     }
 }
     
      
     
 fn upsample(data:Data, out_size:[u16;3], scaling_factor:u16)->Data {
     
    let out_buffer= data.new(out_size);
    let out_channels = out_size[0];
    let out_height = out_size[1];
    let out_width = out_size[2];

    for i in 0..data.height {
       for j in 0..data.width{
          out_buffer[i*scaling_factor][j*scaling_factor]= data[i][j];
          out_buffer[i*scaling_factor+1][j*scaling_factor+1]= data[i][j];
       }
    }
 }        
 
 fn bottleneck(data:Data,c1:i32, c2:i32, shortcut:bool, g:i32, e:f32)-> Data{	// ch_in, ch_out, shortcut, groups, expansion
    let cbomb:i32 =(c2*e);  // hidden channels
    conv(c1, cbomb, 1, 1);
    conv(cbomb, c2, 3, 1, g=g);
    if c1==c2 & shortcut {
      data.add(data);	
    
    } 
 }

  // CSP Bottleneck with 3 convolutions    
 fn C3(c1:i32, c2:i32, shortcut:bool, g:i32, e:f32)-> Data{	// ch_in, ch_out, shortcut, groups, expansion
     
     let cbomb:i32 = (c2 * e);  //hidden channels

     let d1:Data = conv(c1, cbomb, 1, 1);
     bottleneck(cbomb, cbomb, shortcut, g, e=1.0);     //for _ in range(n)
     let d2:Data = conv(c1, cbomb, 1, 1);
     let cdata:Data =concat(d1, d2);
     conv(2 * cbomb, c2, 1, cdata);  //optional act=FReLU(c2)
 }






}