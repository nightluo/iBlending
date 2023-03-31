import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from skimage.io import imsave
from utils import compute_gt_gradient, make_canvas_mask, numpy2tensor, laplacian_filter_tensor, MeanShift, Vgg16, gram_matrix

import os
import utils
from torchvision import transforms

def ib_box(source_file, target_file, output_dir, box):
    ss = 50
    ts = 299
    gpu_id = 6
    num_steps = 1000

    os.makedirs(output_dir, exist_ok = True)

    # Inputs
    source_file = source_file
    target_file = target_file

    # target_img_name = target_img_name
    # target_json_name = target_json_name
    # target_json = os.path.join(target_root, target_json_name)

    # Hyperparameter Inputs
    gpu_id = gpu_id
    num_steps = num_steps
    # source image size
    ss = ss
    # target image size
    ts = ts 

    # get track box
    box = box
    # box = utils.get_points(target_json)



    # Default weights for loss functions in the first pass
    grad_weight = 1e4; style_weight = 1e4; content_weight = 1; tv_weight = 1e-6

    # Load Images
    source_img = Image.open(source_file).convert('RGB')
    sh = source_img.size[0]
    sw = source_img.size[1]
    # imsave(os.path.join(output_dir, 'source_img.png'), source_img)
    # source_img = np.array(Image.open(source_file).convert('RGB').resize((ss, ss)))

    target_img = Image.open(target_file).convert('RGB')
    tw = target_img.size[0]
    th = target_img.size[1]
    # imsave(os.path.join(output_dir, 'target_img.png'), target_img)
    # target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))
    # imsave(os.path.join(output_dir, 'target_img.png'), target_img.astype(np.uint8))
    # print(f"w, h: {sw}, {sh}, {tw}, {th}")

    # blending location
    x_center = th // 2
    y_center = tw // 2

    x_start = box[0]
    x_end = box[1]
    y_start = box[2]
    y_end = box[3]

    # print(f"center, start, end: {x_center}, {y_center}, {x_start}, {x_end}, {y_start}, {y_end}")

    # if x_start < x_center:
    #     x_start += ss // 2
    # else:
    #     x_start -= ss // 2
    # if y_start < y_center:
    #     y_start += ss // 2
    # else:
    #     y_start -= ss // 2

    sh = x_end - x_start
    sw = y_end - y_start
    # print(f"sw, sh -- resize:{sw}, {sh}")

    source_img = Image.open(source_file).convert('RGB').resize((sw, sh))

    mask_img = utils.get_mask(source_file, sh, sw)
    # imsave(os.path.join(output_dir, 'mask_img.png'), mask_img.astype(np.uint8))
    mask_img[mask_img > 0 ] = 1
    # print(f"mask_img:{mask_img}")
    # mask_img = mask_img.resize(sh, sw)
    # print(f"mask_img:{mask_img}")

    # Make Canvas Mask
    canvas_mask = make_canvas_mask(x_start, y_start, x_end, y_end, target_img, mask_img)

    canvas_mask = numpy2tensor(canvas_mask, gpu_id)

    # print(f"canvas_mask:{canvas_mask.shape}")
    canvas_mask = canvas_mask.squeeze(0).repeat(3,1).view(3,th,tw).unsqueeze(0)
    # print(f"canvas_mask:{canvas_mask.shape}")

    # canvas_mask = canvas_mask.squeeze(0).repeat(3,1).view(3,ts,ts).unsqueeze(0)

    target_img = np.array(target_img)
    source_img = np.array(source_img)
    mask_img = np.array(mask_img)

    # Compute Ground-Truth Gradients
    gt_gradient = compute_gt_gradient(x_start, y_start, x_end, y_end, source_img, target_img, mask_img, gpu_id)

    # Convert Numpy Images Into Tensors
    source_img = torch.from_numpy(source_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)
    input_img = torch.randn(target_img.shape).to(gpu_id)

    mask_img = numpy2tensor(mask_img, gpu_id)

    # print(f"mask_img:{mask_img.shape}")
    mask_img = mask_img.squeeze(0).repeat(3,1).view(3,sh,sw).unsqueeze(0)
    # print(f"mask_img:{mask_img.shape}")
    # mask_img = mask_img.squeeze(0).repeat(3,1).view(3,ss,ss).unsqueeze(0)

    # print(f"target_img:{target_img.shape},input_img:{input_img.shape}, canvas_mask:{canvas_mask.shape}")
    resize_299 = transforms.Resize([ts, ts])
    resize_target = transforms.Resize([th, tw])
    # target_img = resize_299(target_img)
    # input_img = resize_299(input_img)
    # canvas_mask = resize_299(canvas_mask)
    # # target_img = target_img.resize((ts, ts))
    # # input_img = input_img.resize((ts, ts))
    # # canvas_mask = canvas_mask.resize((ts, ts))
    # print(f"target_img:{target_img.shape},input_img:{input_img.shape}, canvas_mask:{canvas_mask.shape}")

    # print(f"start, end: {x_start}, {x_end}, {y_start}, {y_end}")
    # x_start = int(x_start * ts / th)
    # x_end = int(x_end * ts / th)
    # y_start = int(y_start * ts / tw)
    # y_end = int(y_end * ts / tw)
    # print(f"start, end: {x_start}, {x_end}, {y_start}, {y_end}")


    # Define LBFGS optimizer
    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    optimizer = get_input_optimizer(input_img)

    # Define Loss Functions
    mse = torch.nn.MSELoss()

    # Import VGG network for computing style and content loss
    mean_shift = MeanShift(gpu_id)
    vgg = Vgg16().to(gpu_id)
    
    run = [0]
    
    # print(f"start, end: {x_start}, {x_end}, {y_start}, {y_end}")
    # x_start_vgg = int(x_start * ts / th)
    # x_end_vgg = int(x_end * ts / th)
    # y_start_vgg = int(y_start * ts / tw)
    # y_end_vgg = int(y_end * ts / tw)
    # print(f"start, end: {x_start}, {x_end}, {y_start}, {y_end}")
    while run[0] <= num_steps:

        def closure():
            # Composite Foreground and Background to Make Blended Image
            blend_img = torch.zeros(target_img.shape).to(gpu_id)
            # print(f"target_img:{target_img.shape}, input_img:{input_img.shape}, canvas_mask:{canvas_mask.shape}, mask_img:{mask_img.shape}")
            blend_img = input_img * canvas_mask + target_img * (canvas_mask - 1) * (-1) 
            
            # Compute Laplacian Gradient of Blended Image
            pred_gradient = laplacian_filter_tensor(blend_img, gpu_id)
            
            # Compute Gradient Loss
            grad_loss = 0
            for c in range(len(pred_gradient)):
                grad_loss += mse(pred_gradient[c], gt_gradient[c])
            grad_loss /= len(pred_gradient)
            grad_loss *= grad_weight
            
            # Compute Style Loss
            target_img_vgg = resize_299(target_img)
            target_features_style = vgg(mean_shift(target_img_vgg))
            # target_features_style = vgg(mean_shift(target_img))
            target_gram_style = [gram_matrix(y) for y in target_features_style]
            
            input_img_vgg = resize_299(input_img)
            blend_features_style = vgg(mean_shift(input_img_vgg))
            # blend_features_style = vgg(mean_shift(input_img))
            blend_gram_style = [gram_matrix(y) for y in blend_features_style]
            
            style_loss = 0
            for layer in range(len(blend_gram_style)):
                style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
            style_loss /= len(blend_gram_style)  
            style_loss *= style_weight           

            
            # Compute Content Loss
            # print(f"blend_img:{blend_img.shape}")
            blend_obj = blend_img[:,:,x_start:x_end, y_start:y_end]

            # blend_obj = blend_img[:,:,int(x_start-source_img.shape[2]*0.5):int(x_start+source_img.shape[2]*0.5), int(y_start-source_img.shape[3]*0.5):int(y_start+source_img.shape[3]*0.5)]
            source_object_features = vgg(mean_shift(source_img*mask_img))
            blend_object_features = vgg(mean_shift(blend_obj*mask_img))
            content_loss = content_weight * mse(blend_object_features.relu2_2, source_object_features.relu2_2)
            content_loss *= content_weight

            # Compute TV Reg Loss
            tv_loss = torch.sum(torch.abs(blend_img[:, :, :, :-1] - blend_img[:, :, :, 1:])) + \
                    torch.sum(torch.abs(blend_img[:, :, :-1, :] - blend_img[:, :, 1:, :]))
            tv_loss *= tv_weight
            
            # Compute Total Loss and Update Image
            loss = grad_loss + style_loss + content_loss + tv_loss
            optimizer.zero_grad()
            loss.backward()

            # Print Loss
            if run[0] % 1 == 0:
                print("run {}:".format(run))
                print('grad : {:4f}, style : {:4f}, content: {:4f}, tv: {:4f}'.format(\
                            grad_loss.item(), \
                            style_loss.item(), \
                            content_loss.item(), \
                            tv_loss.item()
                            ))
                print()
            
            run[0] += 1
            return loss
        
        optimizer.step(closure)

    # clamp the pixels range into 0 ~ 255
    input_img.data.clamp_(0, 255)

    # Make the Final Blended Image
    blend_img = torch.zeros(target_img.shape).to(gpu_id)
    blend_img = input_img*canvas_mask + target_img*(canvas_mask-1)*(-1) 
    blend_img_np = blend_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]

    # Save image from the first pass
    first_pass_img_file = os.path.join(output_dir, 'first_pass.png')
    imsave(first_pass_img_file, blend_img_np.astype(np.uint8))

    ###################################
    ########### Second Pass ###########
    ###################################

    # Default weights for loss functions in the second pass
    style_weight = 1e7; content_weight = 1; tv_weight = 1e-6
    # ss = 512; ts = 512
    num_steps = num_steps

    first_pass_img = np.array(Image.open(first_pass_img_file).convert('RGB'))
    # first_pass_img = np.array(Image.open(first_pass_img_file).convert('RGB').resize((ss, ss)))

    target_img = np.array(Image.open(target_file).convert('RGB'))
    # target_img = np.array(Image.open(target_file).convert('RGB').resize((ts, ts)))

    first_pass_img = torch.from_numpy(first_pass_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)

    target_img = torch.from_numpy(target_img).unsqueeze(0).transpose(1,3).transpose(2,3).float().to(gpu_id)

    # print(f"target_img:{target_img.shape},first_pass_img:{first_pass_img.shape}")

    first_pass_img = first_pass_img.contiguous()

    target_img = target_img.contiguous()

    # print(f"target_img:{target_img.shape},first_pass_img:{first_pass_img.shape}")

    # Define LBFGS optimizer
    def get_input_optimizer(first_pass_img):
        optimizer = optim.LBFGS([first_pass_img.requires_grad_()])
        return optimizer

    optimizer = get_input_optimizer(first_pass_img)

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:
        
        def closure():
            
            # Compute Loss Loss 
            target_img_vgg = resize_299(target_img)
            target_features_style = vgg(mean_shift(target_img_vgg))  
            # target_features_style = vgg(mean_shift(target_img))
            
            target_gram_style = [gram_matrix(y) for y in target_features_style]

            first_pass_img_vgg = resize_299(first_pass_img)
            blend_features_style = vgg(mean_shift(first_pass_img_vgg)) 
            # blend_features_style = vgg(mean_shift(first_pass_img))
            blend_gram_style = [gram_matrix(y) for y in blend_features_style]
            style_loss = 0
            for layer in range(len(blend_gram_style)):
                style_loss += mse(blend_gram_style[layer], target_gram_style[layer])
            style_loss /= len(blend_gram_style)  
            style_loss *= style_weight        
            
            # Compute Content Loss
            first_pass_img_vgg = resize_299(first_pass_img)
            content_features = vgg(mean_shift(first_pass_img_vgg)) 
            # content_features = vgg(mean_shift(first_pass_img))

            content_loss = content_weight * mse(blend_features_style.relu2_2, content_features.relu2_2)
            
            # Compute Total Loss and Update Image
            loss = style_loss + content_loss
            optimizer.zero_grad()
            loss.backward()
            
            # Print Loss
            if run[0] % 1 == 0:
                print("run {}:".format(run))
                print(' style : {:4f}, content: {:4f}'.format(\
                            style_loss.item(), \
                            content_loss.item()
                            ))
                print()
            
            run[0] += 1
            return loss
        
        optimizer.step(closure)

    # clamp the pixels range into 0 ~ 255
    first_pass_img.data.clamp_(0, 255)
    print(f"first_pass_img:{first_pass_img.shape}")
    # Make the Final Blended Image
    input_img_np = first_pass_img.transpose(1,3).transpose(1,2).cpu().data.numpy()[0]
    print(f"input_img:{input_img_np.shape}")
    # input_img_np = resize_target(input_img_np)

    # Save image from the second pass
    imsave(os.path.join(output_dir, 'second_pass.png'), input_img_np.astype(np.uint8))
    print(x_start, x_end, y_start, y_end)
    x1, x2, y1, y2 = x_start, x_end, y_start, y_end
    second_pass_img = np.array(Image.open(first_pass_img_file).convert('RGB').resize((input_img_np.shape[1], input_img_np.shape[0])))
    # x1, x2, y1, y2 = int((x_start-ss*0.5)*512/299), int((x_start+ss*0.5)*512/299), int((y_start-ss*0.5)*512/299), int((y_start+ss*0.5)*512/299)
    second_pass_img[x1:x2, y1:y2] = input_img_np[x1:x2, y1:y2]

    imsave(os.path.join(output_dir, 'second_pass_box.png'), second_pass_img.astype(np.uint8))

