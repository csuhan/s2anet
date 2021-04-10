## Rotated IoU Loss

### Differentiable IoU between Polygons

In general, IoU calculation for rotated boxes [1] is not differentiable as it requires [triangulation](https://en.wikipedia.org/wiki/Triangulation) to calculate the area of intesection polygons, like the [box_iou_rotated](mmdet/ops/box_iou_rotated) op.

[2] proposes an IoU Loss for rotated 2D/3D objects, however, both implementation details and codes are not provided.
[3] shows to calculate IoU between polygons in a pixel-wise manner.  

Recently, we find two implementations for differantiable IoU calculation, [4] and [5].

Here we use the picture in [4] to illustrate this method.
1. Find intesection points of two polygons.
2. Sorting these points to get the indices.
3. Using `torch.gather` to fetch the real values of points.
4. Calculating intesection areas by Shoelace_formula [6].

As the `torch.gather` is differentiable, the whole function is differentiable.

<img src="https://user-images.githubusercontent.com/44202004/96906577-47612500-149a-11eb-82ef-2904800405e0.png" width="60%"></img>


### IoU Loss for Oriented Object Detection

We modify the code in [4] to our codebase and evaluate its performance on DOTA and HRSC2016. Specifically, we add an op called `box_iou_rotated_diff` lies in [here](mmdet/ops/box_iou_rotated_diff).

The results show that optimizing bbox regression with IoU Loss can further boost the performance. Here we list the results in the table below.

|Model                      |Data           |    Backbone     |  reg. loss |  Rotate | Lr schd  | box AP | 
|:-------------:            |:-------------:| :-------------: | :--------: | :-----: | :-----:  | :----: | 
|RetinaNet                  |HRSC2016       |    R-50-FPN     | smooth l1  |   ✓     |   6x     |  81.63 |
|RetinaNet                  |HRSC2016       |    R-50-FPN     |   IoU      |   ✓     |   6x     |  **82.74** |
|CS<sup>2</sup>A-Net-2s     |DOTA           |    R-50-FPN     | smooth l1  |   -     |   1x     |  73.83 |
|CS<sup>2</sup>A-Net-2s     |DOTA           |    R-50-FPN     |   IoU      |   -     |   1x     |  **74.58** |


**References**

[1] Arbitrary-oriented scene text detection via rotation proposals

[2] IoU Loss for 2D/3D Object Detection

[3] PIoU Loss: Towards Accurate Oriented Object Detection in Complex Environments

[4] https://github.com/lilanxiao/Rotated_IoU

[5] https://github.com/maudzung/Complex-YOLOv4-Pytorch

[6] https://en.wikipedia.org/wiki/Shoelace_formula