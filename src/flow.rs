pub fn model (){
  //First Layer
  
   conv(3, 32, kernel_size=(6, 6), stride=(2, 2), padding=(2, 2));
     

//Second Layer

    C3(
        (cv1): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv2): Conv(
        (conv): Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv3): Conv(
        (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (m): Sequential(
        (0): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        )
    )

    //Third Layer

    (3): Conv(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU(inplace=True)
    )

    //Forth Layer

    (4): C3(
        (cv1): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv2): Conv(
        (conv): Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (m): Sequential(
        (0): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        (1): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        )
    )

    //Fifth Layer

    (5): Conv(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU(inplace=True)
    )
    
    //Layer(6)
    C3(
        (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (m): Sequential(
        (0): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        (1): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        (2): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        )
    )
    //Layer(7)
    (7): Conv(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU(inplace=True)
    )
    
    //Layer(8)
    (8): C3(
        (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (m): Sequential(
        (0): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        )
    )
    
    //Layer(9)
    (9): SPPF(
        (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv2): Conv(
        (conv): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (m): MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
    )
    //Layer(10)
    (10): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
    )
    
    //Layer(11)
    (11): Upsample(scale_factor=2.0, mode=nearest)
    
    //Layer(12)
    (12): Concat()
    
    //Layer(13)
    (13): C3(
        (cv1): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv2): Conv(
        (conv): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (m): Sequential(
        (0): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        )
    )
    
    //Layer(14)
    (14): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
    )
    //Layer(15)
    (15): Upsample(scale_factor=2.0, mode=nearest)
    
    //Layer(16)
    (16): Concat()

    //Layer(17)
    (17): C3(
        (cv1): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv2): Conv(
        (conv): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv3): Conv(
        (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (m): Sequential(
        (0): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        )
    )
    
    //Layer(18)
    (18): Conv(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU(inplace=True)
    )
    //Layer(19)
    (19): Concat()

    //Layer(20)
    (20): C3(
        (cv1): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv2): Conv(
        (conv): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv3): Conv(
        (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (m): Sequential(
        (0): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
            )
        )
        )
    )

    //Layer(21)
    (21): Conv(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (act): SiLU(inplace=True)
    )

    //Layer(22)
    (22): Concat()
    
    //Layer(23)
    (23): C3(
        (cv1): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv2): Conv(
        (conv): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (cv3): Conv(
        (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
        (act): SiLU(inplace=True)
        )
        (m): Sequential(
        (0): Bottleneck(
            (cv1): Conv(
            (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
            (act): SiLU(inplace=True)
            )
            (cv2): Conv(
            (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (act): SiLU(inplace=True)
              )
            )
          )
        )

//Layer(24)
        (24): Detect(
          (m): ModuleList(
            (0): Conv2d(128, 255, kernel_size=(1, 1), stride=(1, 1))
            (1): Conv2d(256, 255, kernel_size=(1, 1), stride=(1, 1))
            (2): Conv2d(512, 255, kernel_size=(1, 1), stride=(1, 1))
          )
        )
      
    
  

}

