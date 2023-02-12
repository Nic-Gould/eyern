using lzzz:

struct data 
    {
	depth: u16, 		
	width:u16, 
	height: u16, 
	
	pixel: f16,
	row:arr[pixel;width],
	frame:arr[row;height],
	channels:arr[frame; depth],
    }
impl data 
    {
	fn new(&self, size: arr[i32,i32,i32])->self
        {
		depth 	= size[0];	
		width 	= size[1];
		height 	= size[2];

		pixel: f16;
		row: arr = [pixel: width];
		frame: arr = [row: height];
		channels: arr = [frame: depth];
	    }
    fn conv(self,[layer_num, out_size, params])->data
        {
        padding[in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1]
        multipy_buffer= data.new(out_size);
        out_channels = out_size[0];
        out_height = out_size[1];
        out_width = out_size[2];
        tensor_buffer = lzzz:decompress(layer_num) 
        
        for a in out_channels
		    total = 0;
	        count = 0;
	
            for b in in_channels
                {
                for i in num_strides_high
                    {
                    for j in num_strides_wide
                        {
                        for l in ksize
                            {
                            for k in ksize
                                {
                                multiply_buffer[a][i][j] += data[b][(i*s) + (k*d)-1] [(j*s)+(l*d)-1] * weights[a][b][l][k];
                                total += multiply_buffer[a][i][j];
                                count +=1;
                                }
                            }
                        }
                    }
                }
            avg = total /count;
            sum_var_sq = 0;
            count = 0;
            
            for i in out_height
                {
                for j in out_width
                    {
                    multiply_buffer[i][j] -= avg;
                    sum_var_sq += (multiply_buffer[i][j] * multiply_buffer[i][j]);
                    count +=1;
                    }
                }
            std_dev = (sum_var_sq/count)^0.5;
            
            for i in out_height
                {
                for j in out_width
                    {
                    multiply_buffer[i][j] /= std_dev;
                    1/(1+e^(-(multiply_buffer[i][j]));
                    }
                }
                    
            }
        }
    fn maxpool(&mut data, ksize, strides = ksize, padding)
        {    
        for b in in_channels
            {
            for i in num_strides_high
                {
                for j in num_strides_wide
                    {
                    pool_max=0;
                    for l in ksize
                        {
                        for k in ksize
                            {
                            if data[b][(i*s) + (k*d)] [(j*s)+(l*d)] > pool_max:
                                {
                                pool_max = data[b][(i*s) + (k*d)] [(j*s)+(l*d)];
                                }
                            }
                        }
                    }
                }
            }
        }
    fn concat(data: M1, data: M2, i32: out_size, i32: data_len)->data{
        concat_buffer= data.new(out_size);
        //as with all other parameters, data len is precalculated.
        for row in concat_buffer{
            row [:data_len]= M1[:];
            row[data_len:] = M2[:];
        }
    }

    fn bottleneck(data, c1, c2, shortcut=True, g=1, e=0.5)->data {  // ch_in, ch_out, shortcut, groups, expansion
        if shortcut and c1==c2:
        {
        bypass=data.clone();
        }   
        c_=int(c2*e)  # hidden channels
        data.Conv(c1, c_, 1, 1)
        data.Conv(c_, c2, 3, 1, g=g)
        if bypass:
            data + clone;    
    }

    fn upsample(&self, out_size, scaling_factor)->data{
        out_buffer= data.new(out_size);
        out_channels = out_size[0];
        out_height = out_size[1];
        out_width = out_size[2];

        for i in data.height
            for j in data.width
                out_buffer[i*scaling_factor][j*scaling_factor]= data[i][j]
                out_buffer[i*scaling_factor+1][j*scaling_factor+1]= data[i][j]
    }
        
    fn C3(self, c1, c2, n=1, shortcut=True, g=1, e=0.5)->data{  //ch_in, ch_out, number, shortcut, groups, expansion
        // CSP Bottleneck with 3 convolutions
        d2 = self.clone; 
        c_ = int(c2 * e);  //hidden channels
        data.Conv(c1, c_, 1, 1);
        data.Bottleneck(c_, c_, shortcut, g, e=1.0)     //for _ in range(n)
        
        d2.Conv(c1, c_, 1, 1)
        cdata=concat(data, d2);
        cdata.Conv(2 * c_, c2, 1)  //optional act=FReLU(c2)
    }
    struct Data 
    {
	depth: u16, 		
	width:u16, 
	height: u16, 
	
	pixel: f32,
	row:[pixel;width],
	frame:[row;height],
	channels:[frame; depth],
    }
impl data 
    {
	fn new(&self, size: [i32;3])->self
        {
		depth 	= size[0];	
		width 	= size[1];
		height 	= size[2];

		row = [pixel, width];
		frame = [row, height];
		channels = [frame, depth];
	    }
    
        
        let &mut multiply buffer = toTensor(,Hout,Wout, empty)
        let num_strides_high = channel.shape[0]/s
		let num_strides_wide = channel.shape[1]/s
    
    
    
    
    
    
    
    
    
        //[#derive_default]   
    
    




	fn conv(in_channels:i32, out_channels:i32, ksize:i32, s:i32, p:i32, d:i32, weights:[[[[f32;i32];i32];i32];i32]) -> data
		{
        let out_height = ((input.size[1]+2*p-d*(ksize-1) )-1/s)+1;
        let out_width = ((input.size[2]+2*p-d*(ksize-1) )-1/s)+1;
        let out_size = out_channels * out_height * out_width;
		let mut multipy_buffer= Data::new(out_size);
        for a in out_channels
			{
			let mut total = 0;
			let mut count = 0;
			for b in in_channels
				{
				for i in num_strides_high
					{
					for j in num_strides_wide
						{
						for l in ksize
							{
							for k in ksize
								{
								multiply_buffer[a][i][j] += data[b][(i*s) + (k*d)-1] [(j*s)+(l*d)-1] * weights[a][b][l][k];
								total += multiply_buffer[a][i][j];
								count +=1;
								}
							}
						}
					}
				}
			let avg = total /count;
			let mut sum_var_sq = 0;
			count = 0;
			
			for i in out_height
				{
				for j in out_width
					{
					multiply_buffer[i][j] -= avg;
					sum_var_sq += (multiply_buffer[i][j] * multiply_buffer[i][j]);
					count +=1;
					}
				}
			std_dev = (sum_var_sq/count)^0.5;
			
			for i in out_height
				{
				for j in out_width
					{
					multiply_buffer[i][j] /= std_dev;
					sigmoid(multiply_buffer[i][j]);
					}
				}
					
			}
		}
	fn sigmoid(x: &f32)->f32
		{
		1/(1+e^(-x))
		}
	
	fn bottleneck(data, c1, c2, shortcut=True, g=1, e=0.5)	// ch_in, ch_out, shortcut, groups, expansion
		{  
		let cbomb:i32 =(c2*e);  // hidden channels
		self.Conv(c1, cbomb, 1, 1);
		self.Conv(cbomb, c2, 3, 1, g=g);
		if c1==c2 & shortcut
			{
			self.add(data);	
			} 
		}
        


        
    fn conv(self,[layer_num, out_size, params])->data
        {
        padding[in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1]
        multipy_buffer= data.new(out_size);
        out_channels = out_size[0];
        out_height = out_size[1];
        out_width = out_size[2];
        tensor_buffer = lzzz:decompress(layer_num) 
        
        for a in out_channels
            {
            total = 0;
            count = 0;

            for b in in_channels
                {
                for i in num_strides_high
                    {
                    for j in num_strides_wide
                        {
                        for l in ksize
                            {
                            for k in ksize
                                {
                                multiply_buffer[a][i][j] += data[b][(i*s) + (k*d)-1] [(j*s)+(l*d)-1] * weights[a][b][l][k];
                                total += multiply_buffer[a][i][j];
                                count +=1;
                                }
                            }
                        }
                    }
                }
            avg = total /count;
            sum_var_sq = 0;
            count = 0;
            
            for i in out_height
                {
                for j in out_width
                    {
                    multiply_buffer[i][j] -= avg;
                    sum_var_sq += (multiply_buffer[i][j] * multiply_buffer[i][j]);
                    count +=1;
                    }
                }
            std_dev = (sum_var_sq/count)^0.5;
            
            for i in out_height
                {
                for j in out_width
                    {
                    multiply_buffer[i][j] /= std_dev;
                    1/(1+e^(-(multiply_buffer[i][j]));
                    }
                }                 
        
            }
        }        
    fn maxpool(&mut data, ksize, strides = ksize, padding)
        {    
        for b in in_channels
            {
            for i in num_strides_high
                {
                for j in num_strides_wide
                    {
                    pool_max=0;
                    for l in ksize
                        {
                        for k in ksize
                            {
                            if data[b][(i*s) + (k*d)] [(j*s)+(l*d)] > pool_max:
                                {
                                pool_max = data[b][(i*s) + (k*d)] [(j*s)+(l*d)];
                                }
                            }
                        }
                    }
                }
            }
        }
    fn concat(data: M1, data: M2, i32: out_size, i32: data_len)->data{
        concat_buffer= data.new(out_size);
        //as with all other parameters, data len is precalculated.
        for row in concat_buffer
            {
            row [:data_len]= M1[:];
            row[data_len:] = M2[:];
            }
        }
        
         
        
    fn upsample(&self, out_size, scaling_factor)->data
        {
        out_buffer= data.new(out_size);
        out_channels = out_size[0];
        out_height = out_size[1];
        out_width = out_size[2];

        for i in data.height()
            {
            for j in data.width
                {
                out_buffer[i*scaling_factor][j*scaling_factor]= data[i][j]
                out_buffer[i*scaling_factor+1][j*scaling_factor+1]= data[i][j]
                }
            }
        }        
            
     // CSP Bottleneck with 3 convolutions    
    fn C3(self, c1, c2, n=1, shortcut=True, g=1, e=0.5)-> data  //ch_in, ch_out, number, shortcut, groups, expansion
        {
        d2 = self.clone; 
        c = int(c2 * e);  //hidden channels
        data.Conv(c1, c_, 1, 1);
        data.Bottleneck(c_, c_, shortcut, g, e=1.0)     //for _ in range(n)
        
        d2.Conv(c1, c_, 1, 1)
        cdata=concat(data, d2);
        cdata.Conv(2 * c_, c2, 1)  //optional act=FReLU(c2)
        }
	
	}