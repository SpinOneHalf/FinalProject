import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import islice



def animate_images(list_images,path="temp.mp4"):
    velocity=list_images[0]
    images=list_images[1]
    fig,axis=plt.subplots(2,1)
    space_plane=axis[0].imshow(format_tensor_image(images[0]))
    velocity_plane=axis[1].imshow(format_tensor_image(velocity[0]))
    def animation_func(i):
        space_plane.set_array(format_tensor_image(images[i]))
        velocity_plane.set_array(format_tensor_image(velocity[i]))
        return space_plane,velocity_plane
    anim=animation.FuncAnimation(fig,animation_func,frames=len(images),interval=1000/len(images))
    return anim
def format_tensor_image(tensor_image):
    return tensor_image.permute(1,2,0)