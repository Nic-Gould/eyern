use std::str::FromStr;
use std::fs;

pub fn main(){

let file_path = "model.0.conv.weight";
load_weights(file_path);



fn load_weights(file_path: &str)-> Vec<f32>
   {
   let trimmybois: &[_] = &['[', ']'];
   let bump: Vec<f32> = fs::read_to_string(file_path)
                        .expect("Should have been able to read the file")
                        .replace(trimmybois, "")
                        .split_whitespace()
                        .filter_map(|s| s.parse::<f32>().ok())
                        .collect();
   println!("{}", bump.len());
   bump
   }
   
}

   





//let arr: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 4]>> = ndarray_npy::read_npy(reader).unwrap();
//let arr = ndarray_npy::read_npy(reader);
