fn get_esp32()->data

//From <https://github.com/espressif/esp32-camera/blob/master/driver/include/esp_camera.h> 
* @brief Obtain pointer to a frame buffer.	
	 *
	 * @return pointer to the frame buffer
	 */
	
camera_fb_t* esp_camera_fb_get();

data= data.new((resolution[frame_size].width, resolution[frame_size].height, 3);

for i,j,k in data.channels, data.frames, data.rows{
	channels[i][j][k] = &fb[i][j][k];		
	}


* @brief Return the frame buffer to be reused again.

 *

 * @param fb Pointer to the frame buffer

 */

Void esp_camera_fb_return(camera_fb_t* fb);


Fn get_image(path: string)-> data{

    File = open(path)
    input= data.new(file[0])
    For I,j,k in input.channels, input.frames, input.rows{
        Channels[i][j][k] = &fb[i][j][k]		// reference faster than copy?
        }

_______________________________________________________________
