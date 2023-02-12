

let d1 = layer1(params, weights,prev)
let d2= layer2(params, weights,prev)

 

fn layer2(params, weights, d1)->Data{

            let d2 = Data::new([size]);       //shouldn't conflict due to variable shaddowing.

            conv(params,weights,d1,d2);             //actually conv should have access to these from the scope right?

            drop(d1);

            d2                                            // returns a data type matching the declarataion in the outer scope.

 

}




//First Layer
conv(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2))

//Second Layer
Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

//Third Layer
Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

//Forth Layer
Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

//Fifth Layer
Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

//Layer(6)
Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

//Layer(7)
Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

//Layer(8)
Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

//Layer(9)
Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)

//Layer(10)
Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))

//Layer(11)
Upsample(scale_factor=2.0, mode=nearest)

//Layer(12)
Concat()

//Layer(13)
Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

//Layer(14)
Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))

//Layer(15)
Upsample(scale_factor=2.0, mode=nearest)

//Layer(16)
Concat()

//Layer(17)
Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

//Layer(18)
Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

//Layer(19)
Concat()

//Layer(20)
Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

//Layer(21)
Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
//Layer(22)

Concat()

//Layer(23)
Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

//Layer(24)
Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
