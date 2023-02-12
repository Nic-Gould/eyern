
   //[#derive_default]   
   fn conv(in_channels:u16, out_channels:u16, ksize:u16, s:u16, p:u16, d:u16, weights:Data, data:Data) -> Data{
      
      let out_height = ((data.height+2*p-d*(ksize-1) )-1/s)+1;
      let out_width = ((data.width+2*p-d*(ksize-1) )-1/s)+1;
      let out_size:u16 = out_channels * out_height * out_width;
      let num_strides_high = data.height/s;
      let num_strides_wide = data.width/s;
      let mut multiply_buffer= Data::new(out_size);
      for a in 1..out_channels{
         let mut total = 0;
         let mut count = 0;
         for b in 1..in_channels{
            for i in 1..num_strides_high{
               for j in 1..num_strides_wide{
                  for l in 1..ksize{
                     for k in 1..ksize{
                        multiply_buffer[a][i][j] += data[b][(i*s) + (k*d)-1] [(j*s)+(l*d)-1] * weights[a][b][l][k];
                        total += multiply_buffer[a][i][j];
                        count +=1;
                     }
                  }
               }
            }
         }
         multiply_buffer
      
   }
}
 
       
  
   
   fn maxpool(&mut data:Data, ksize:u16, s:u16 , p:u16) {    
      for b in 1..in_channels {
         for i in 1..num_strides_high {
            for j in 1..num_strides_wide {
               let pool_max=0;
               for l in 1..ksize {
                  for k in 1..ksize {
                     if data[b][(i*s) + (k*d)] [(j*s)+(l*d)] > pool_max {
                        pool_max = data[b][(i*s) + (k*d)] [(j*s)+(l*d)];
                     }
                  }
               }
            }
         }
      }
   }
   
   fn concat(data1: Data, data2: Data, out_size:[u16;3], data_len:i32)->Data{
      let mut concat_buffer= Data::new(out_size);
       //as with all other parameters, data len is precalculated.
      for row in concat_buffer.iter() {
         row[;data_len] = data1[:];
         row[data_len:] = data2[:];
      } concat_buffer
   }
       
        
       
   fn upsample(&self, out_size[u13;3], scaling_factor:u16)->data {
       
      let out_buffer= data.new(out_size);
      let out_channels = out_size[0];
      let out_height = out_size[1];
      let out_width = out_size[2];

      for i in data.height() {
         for j in data.width(){
            out_buffer[i*scaling_factor][j*scaling_factor]= data[i][j];
            out_buffer[i*scaling_factor+1][j*scaling_factor+1]= data[i][j];
         }
      }
   }        
   
   fn bottleneck(&self, c1:i32, c2:i32, shortcut:bool, g:i32, e:f32)-> Data{	// ch_in, ch_out, shortcut, groups, expansion
      let cbomb:i32 =(c2*e);  // hidden channels
      self.conv(c1, cbomb, 1, 1);
      self.conv(cbomb, c2, 3, 1, g=g);
      if c1==c2 & shortcut {
        self.add(data);	
      self
      } 
   }

    // CSP Bottleneck with 3 convolutions    
   fn C3(&mut self, c1:i32, c2:i32, shortcut:bool, g:i32, e:f32)-> Data{	// ch_in, ch_out, shortcut, groups, expansion
       
       let cbomb:i32 = (c2 * e);  //hidden channels

       let d1:Data = conv(c1, cbomb, 1, 1);
       bottleneck(cbomb, cbomb, shortcut, g, e=1.0);     //for _ in range(n)
       let d2:Data = conv(c1, cbomb, 1, 1);
       let cdata:Data =concat(d1, d2);
       cdata.conv(2 * cbomb, c2, 1);  //optional act=FReLU(c2)
   }
  
}
