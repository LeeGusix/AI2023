import onnx
import onnxruntime
import numpy as np

from PIL import Image 

from torchvision import transforms

from utils.utils import *

import cv2

image_path = "data/samples/11_256.jpg"

#load onnx [success]---------------------------
onnx_model = onnx.load("weights/best_403food_e200b150v2.onnx")
print(onnx.checker.check_model(onnx_model))
#print(onnx.helper.printable_graph(onnx_model.graph))
#ONNX Runtime Test -------------------
ort_session = onnxruntime.InferenceSession("weights/best_403food_e200b150v2.onnx")

to_tensor = transforms.ToTensor()
img = Image.open("data/samples/11_256.jpg")
ort_inputs = {ort_session.get_inputs()[0].name:[to_tensor(img)]}

ort_classe, ort_boxes = ort_session.run(None, ort_inputs)

print(type(to_tensor))
#np.testing.assert_allclose(to_tensor.detach().cpu().numpy(), ort_classe[0], rtol=1e-03, atol=1e-05)

def non_max_suppression(boxes, scores, threshold):
    indices = cv2.dnn.NMSBoxes(boxes, scores, threshold, 0.4)
    result_boxes = [boxes[i] for i in indices]
    result_scores = [scores[i] for i in indices]
    return result_boxes, result_scores



# Visualize results on the original image
original_image = cv2.imread(image_path)
for class_probs, box in zip(ort_classe, ort_boxes):
    # Select the class with the highest probability
    class_index = np.argmax(class_probs)
    confidence = class_probs[class_index]

    # Get box coordinates
    x, y, w, h = box *256

    # Draw bounding box
    color = (0, 255, 0)  # Green color
    thickness = 2
    cv2.rectangle(original_image, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), color, thickness)

    # Display class label and confidence
    label = f'Class: {class_index}, Confidence: {confidence:.2f}'
    print(f"confidence : {confidence} :: class_index : {class_index} :Label : {label}")
    cv2.putText(original_image, label, (int(x - w/2), int(y - h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)


print(len(ort_boxes))
print(len(ort_boxes[0]))

# 예측치
pc = ort_boxes[0][0]
#바운딩 박스
bx,by,bh,bw = ort_boxes[1:5][0]
#모든 예측 가중치
c =ort_boxes

score = (pc *c).argmax()

print(f"값 : {score}\n{ort_boxes.shape} \nXYHW {bx} ,{by} ,{bh} ,{bw} : {c.shape} \n:PC {pc}::{len(c)} \n Score {score}")
    
# for out in ort_boxes:
#     pc = out
#     bx,by,bh,bw = ort_boxes[1:5][0]
#     c =ort_boxes[5:-1][0]

#     score = (pc *c).argmax()

#     print(f"값 : {score}\n{out.shape} \n {bx} ,{by} ,{bh} ,{bw} : {c.shape} : {pc}::{len(out)}")


# color = (0, 255, 0)
# thickness = 2
# original_image = cv2.imread(image_path)
# x, y, w, h = ort_boxes *256
# cv2.putText(original_image, "label", (int(x - w/2), int(y - h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
# # Display the image with bounding boxes
# cv2.imshow('YOLOv3 Object Detection', original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# # 모델 변환
# torch.onnx.export(torch_model,               # 실행될 모델
#                   x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
#                   "super_resolution.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
#                   export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
#                   opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
#                   do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
#                   input_names = ['input'],   # 모델의 입력값을 가리키는 이름
#                   output_names = ['output'], # 모델의 출력값을 가리키는 이름
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
#                                 'output' : {0 : 'batch_size'}})
# torch.onnx.export(model, img, f, verbose=False, opset_version=11,
#                   input_names=['images'], 
#                   output_names=['classes', 'boxes'])

#output_classe = {ort_session.get_outputs()[0] : classes}

# #np.testing.assert_allclose(onnx_model.detach().cpu().numpy(), ort_classe, rtol=1e-03, atol=1e-05)

# #print(f"Prob : {prob} :: ID : {class_id}")


# # Convert box coordinates from (x, y, w, h) to (x1, y1, x2, y2)
# ort_boxes[:, :2] -= ort_boxes[:, 2:] / 2
# ort_boxes[:, 2:] += ort_boxes[:, :2]

# # Run NMS
# confidence_threshold = 0.5  # Adjust as needed
# nms_boxes, nms_scores = non_max_suppression(ort_boxes.tolist(), ort_classe.flatten().tolist(), confidence_threshold)

# # Visualize results on the original image
# original_image = cv2.imread(image_path)
# for box, score in zip(nms_boxes, nms_scores):
#     x1, y1, x2, y2 = box
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

#     # Draw bounding box
#     color = (0, 255, 0)  # Green color
#     thickness = 2
#     cv2.rectangle(original_image, (x1, y1), (x2, y2), color, thickness)

#     # Display class label and confidence
#     label = f'Confidence: {score:.2f}'
#     cv2.putText(original_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

# # Display the image with bounding boxes after NMS
# cv2.imshow('YOLOv3 Object Detection with NMS', original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




















# print(f" ---------Classe[{len(ort_classe)}][{ort_classe.shape}]-------- \n {ort_classe} : ")
# print(f" ---------Boxes[{len(ort_boxes)}][{ort_boxes.shape}]-------- \n {ort_boxes}")

# arr = ort_classe.flatten()
# arr = np.unique(arr)
# print(f"합 : {np.sum(arr)}")

# arrMax = np.argmax(ort_classe, axis=-1)
# print(np.sum(ort_classe))
# print(arrMax[-2])


# # 배열에서 가장 많이 나온 값을 찾음
# most_common_value = np.bincount(arrMax).argmax()

# # 가장 많이 나온 값의 모든 인덱스를 찾음
# indices = np.where(arrMax == most_common_value)[0]
# print(f"가장 많이 나온 값 : {most_common_value}")

# for _ort_classe, _ort_boxes in zip(ort_classe, ort_boxes):
#     # 클래스 정보는 클래스 확률이 가장 높은 인덱스로 결정됨
#     class_index = np.argmax(_ort_classe)
    
#     print(f"Predicted Class Index: {class_index}")


# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# # ONNX 런타임에서 계산된 결과값
# ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)

# # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

# print("Exported model has been tested with ONNXRuntime, and the result looks good!")