import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ConvertImageDtype, Normalize
from simvp_model import SimVP_Model
import numpy as np
import matplotlib.pyplot as plt
from VideoDataset import VideoDataset
from itertools import islice
import torchvision.transforms.functional as tv
import torchmetrics
import mask_r_cnn.pytorch_mask_rcnn as pmr

def unnormalize(tensor, mean, std):
    """Revert the normalization process."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and add the mean
    return tensor

def main():

    # Calculated from the training + unlabeled data set
    mean = torch.tensor([0.5072, 0.5056, 0.5021])
    std = torch.tensor([0.0547, 0.0544, 0.0587])

    transforms = Compose([
            ConvertImageDtype(torch.float),
            Normalize(mean=mean, std=std)
        ])

    eval_dataset = VideoDataset(root_dirs=['/home/charles/PycharmProjects/MachineLearningHomework/Dataset_Student/val'], transform=transforms, sequence_length=11)
    eval_loader = DataLoader(eval_dataset, batch_size=1, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SimVP_Model(in_shape=(11, 3, 240, 160), N_T=6, N_S=4, hid_S=64, hid_T=384, spatio_kernel_enc=3, spatio_kernel_dec=3, drop_path=0.1, drop=0.1)
    model.to(device)
    model.load_state_dict(torch.load('pred_model_mid_sgd_200epoch_checkpoint.pth'))
    model.eval()
    checkpoint = torch.load('maskrcnn_coco-6-17(1).pth')
    num_classes = 49
    mask_network = pmr.maskrcnn_resnet50(False, num_classes, False).to(device)
    mask_network.load_state_dict(checkpoint['model'])
    mask_network.eval()
    jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
    running_iou=0
    with torch.no_grad():
        for inputs,last_frame,tru_mask in eval_loader:
            inputs = inputs.to(device)
            predictions = model(inputs)
            predictions = unnormalize(predictions.cuda(), mean, std)
            last_frame_predicted = predictions[0, -1]
            last_frame_predicted = tv.convert_image_dtype(last_frame_predicted,dtype=torch.uint8)
            mask_stuff=mask_network(last_frame_predicted.float())
            res = mask_stuff
            print(mask_stuff["scores"])
            output=torch.zeros(160,240,dtype=torch.long)
            for i,mask in enumerate(mask_stuff["masks"]):
                indx=mask>.5
                output[indx]=mask_stuff["labels"][i]
            running_iou+=jaccard(output,tru_mask[0])
    print(f"VAL LOSS {running_iou/len(eval_loader):}")
    plt.imshow(output)
    plt.figure(2)
    plt.imshow(tru_mask[0])

    # Unnormalize if necessary
    predictions = unnormalize(predictions.cuda(), mean, std)
    #targets = unnormalize(targets.cpu(), mean, std)

    # Choose a frame from the sequence to display
    last_frame_predicted = predictions[0, -1].permute(1, 2, 0).numpy()
    last_frame_predicted = np.clip(last_frame_predicted, 0, 1)

    #last_frame_actual = targets[0, -1].permute(1, 2, 0).numpy()
    #last_frame_actual = np.clip(last_frame_actual, 0, 1)

    twelve_frame_predicted = predictions[0, 0].permute(1, 2, 0).numpy()
    twelve_frame_predicted = np.clip(twelve_frame_predicted, 0, 1)

    #twelve_frame_actual = targets[0, 0].permute(1, 2, 0).numpy()
    #twelve_frame_actual = np.clip(twelve_frame_actual, 0, 1)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(twelve_frame_predicted)
    ax[0].set_title('Predicted 12th Frame')
    ax[0].axis('off')

    #ax[1].imshow(twelve_frame_actual)
    ax[1].set_title('Actual 12th Frame')
    ax[1].axis('off')

    ax[2].imshow(last_frame_predicted)
    ax[2].set_title('Predicted 22nd Frame')
    ax[2].axis('off')

    #ax[3].imshow(last_frame_actual)
    ax[3].set_title('Actual 22nd Frame')
    ax[3].axis('off')

    plt.show()

if __name__ == '__main__':
    main()