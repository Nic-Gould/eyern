use std::env;
use std::fs;

    //THIS WORKS!!!
    let gread:Vec<f32>  = std::fs::read_to_string("./weights.txt")
    .expect("poopoo")
    .split(&[',',']','[','(',')'][..])
    .filter_map(|s| s.parse::<f32>().ok())
    .collect();

    println!("{}", gread.len());



   //this also works
    
    let butthole: dfdx::tensor::Tensor3D<3,6,6> = dfdx::tensor::TensorCreator::new([[[ 0.00059,  0.02020,  0.02119,  0.04584,  0.02594,  0.02266],
        [ 0.01506,  0.03918,  0.04468,  0.06055,  0.03915,  0.02689],
        [ 0.00964,  0.03775,  0.05939,  0.07440,  0.03177,  0.01109],
        [ 0.01255,  0.03015,  0.04630,  0.05533,  0.00922, -0.00630],
        [-0.02345, -0.01231, -0.00673, -0.00222, -0.03867, -0.05048],
        [-0.04367, -0.04343, -0.05286, -0.03766, -0.05722, -0.05414]],
  
       [[-0.00401,  0.00234, -0.00456,  0.01656,  0.00706,  0.01816],
        [ 0.01523,  0.02922,  0.02527,  0.03616,  0.01949,  0.02039],
        [ 0.01782,  0.03589,  0.04727,  0.05457,  0.01432,  0.00556],
        [ 0.02121,  0.02669,  0.03177,  0.03488, -0.01224, -0.01358],
        [-0.01299, -0.01248, -0.01581, -0.01611, -0.05548, -0.05762],
        [-0.01451, -0.02258, -0.03693, -0.02161, -0.04767, -0.04074]],
  
       [[ 0.00461,  0.01569,  0.01205,  0.03001,  0.01715,  0.02118],
        [ 0.01820,  0.03430,  0.03168,  0.03897,  0.02003,  0.01787],
        [ 0.01562,  0.03253,  0.04507,  0.04715,  0.01041,  0.00062],
        [ 0.01968,  0.02174,  0.02649,  0.02826, -0.01403, -0.01233],
        [-0.00871, -0.01098, -0.01228, -0.01246, -0.04218, -0.04227],
        [-0.00202, -0.01068, -0.02135, -0.00505, -0.02666, -0.01981]]]); 


fn main() {
    let file_path = "weights/weights.txt";
    let contents = fs::read_to_string(file_path)
        .expect("Should have been able to read the file");
        
        fn conv(self: &Self, in_channels:i32, out_channels:i32, ksize:i32, s:i32, p:i32, d:i32, weights:Data) -> Data{
            input.size;
            self.ch   Self nnel.shape;
         
            let out_height = ((input.size[1]+2*p-d*(ksize-1) )-1/s)+1;
            let out_width = ((input.size[2]+2*p-d*(ksize-1) )-1/s)+1;
            let out_size = out_channels * out_height * out_width;
            let num_strides_high = channel.shape[0]/s;
            let num_strides_wide = channel.shape[1]/s;
            let mut multiply_buffer= Data::new(out_size);
            for a in out_channels{
               let mut total = 0;
               let mut count = 0;
               for b in in_channels{
                  for i in num_strides_high{
                     for j in num_strides_wide{
                        for l in ksize{
                           for k in ksize{
                              multiply_buffer[a][i][j] += data[b][(i*s) + (k*d)-1] [(j*s)+(l*d)-1] * weights[a][b][l][k];
                              total += multiply_buffer[a][i][j];
                              count +=1;
                           }
                        }
                     }
                  }
               }
               multiply_buffer
            // The following section applies batch normalisation. I can likely fuse the data and skip this section. Fused data works up to 2x faster in some benchmarks
            // Also, BN for detection uses the running mean, var etc from training and doesn't need to be re-calculated
            
              let avg = total /count;     
              let mut sum_var_sq = 0;
              count = 0;
              
               for i in out_height {
                  for j in out_width {
                     multiply_buffer[i][j] -= avg;
                     sum_var_sq += (multiply_buffer[i][j] * multiply_buffer[i][j]);
                     count +=1;
                  }
               }
              std_dev = (sum_var_sq/count)^0.5;
              
               for i in out_height{
                  for j in out_width{
                    multiply_buffer[i][j] /= std_dev;
                    sigmoid(multiply_buffer[i][j]);
                  }
               }
                    
            }
         }
         
         fn sigmoid(x: &f32)->f32{
           1/(1+e^(-x))
         }

 for char in bump{
   match char {
       "[" => ,
       "]" => ,
       0..9 => ,
       "." => ,
       " " => ,
       _ => , 

   }
}
