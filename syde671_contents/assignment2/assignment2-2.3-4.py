import cv2
import numpy as np

apple = cv2.imread("[INPUT HERE]").astype(np.float32)
orange = cv2.imread("[INPUT HERE]").astype(np.float32)
mask = cv2.imread("[INPUT HERE]").astype(np.float32)

def gaussian_stack(image, levels, kernel_size):
    stack = [image]
    for _ in range(levels - 1):
        stack.append(cv2.GaussianBlur(stack[-1], kernel_size, 0))
    return stack


def laplacian_stack(gaussian):
    stack = []
    for i in range(len(gaussian) - 1):
        stack.append(gaussian[i] - gaussian[i + 1])
    stack.append(gaussian[-1])
    return stack


apple_g = gaussian_stack(apple, 5, (15,15))
orange_g = gaussian_stack(orange, 5, (15,15))
mask_g = gaussian_stack(mask, 5, (301, 301))

apple_l = laplacian_stack(apple_g)
orange_l = laplacian_stack(orange_g)

stacks = [orange_g, apple_g, orange_l, apple_l]

blended_l = []
apple_decomposed_sum = np.zeros_like(apple)
orange_decomposed_sum = np.zeros_like(orange)

grid_rows = {}

for i in range(5):
    apple_comp = apple_l[i] * mask_g[i]
    orange_comp = orange_l[i] * (1.0 - mask_g[i])
    blend_layer = apple_comp + orange_comp
    blended_l.append(blend_layer)
    
    apple_decomposed_sum += apple_comp
    orange_decomposed_sum += orange_comp
    
    if i in [0, 2, 4]:

        offset = 128.0 if i < 4 else 0.0
        factor = 4.0 if i < 4 else 1.0 # visibility purposes
        
        vis_a = np.clip(offset + (apple_comp * factor), 0, 255).astype(np.uint8)
        vis_o = np.clip(offset + (orange_comp * factor), 0, 255).astype(np.uint8)
        vis_b = np.clip(offset + (blend_layer * factor), 0, 255).astype(np.uint8)
        
        # aesthetic clipping purposes
        vis_a[0, :] = vis_a[1, :]
        vis_o[0, :] = vis_o[1, :]
        vis_b[0, :] = vis_b[1, :]
        
        grid_rows[f"row_{i}"] = np.hstack([vis_a, vis_o, vis_b])

sum_a = np.clip(apple_decomposed_sum, 0, 255).astype(np.uint8)
sum_o = np.clip(orange_decomposed_sum, 0, 255).astype(np.uint8)
sum_b = np.clip(np.sum(blended_l, axis=0), 0, 255).astype(np.uint8)

sum_a[0, :] = sum_a[1, :]
sum_o[0, :] = sum_o[1, :]
sum_b[0, :] = sum_b[1, :]

grid_rows["row_sum"] = np.hstack([sum_a, sum_o, sum_b])

full_matrix_grid = np.vstack([
    grid_rows["row_0"],   
    grid_rows["row_2"],  
    grid_rows["row_4"], 
    grid_rows["row_sum"] 
])

cv2.imwrite("[OUTPUT HERE]", full_matrix_grid)
cv2.imwrite("[OUTPUT HERE]", sum_b)