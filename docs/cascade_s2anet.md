## Cascade S<sup>2</sup>A-Net

Cascade S<sup>2</sup>A-Net (CS<sup>2</sup>A-Net) can be regarded as a generalized S<sup>2</sup>A-Net. 
It consists of several detection heads, and each head contains an Alignment Convolution and two subnetworks, _.i.e_, _cls_ and _reg_.
The overall archetecture of CS<sup>2</sup>A-Net is shown below.

![](../demo/cascade_s2anet.png)

The difference between CS<sup>2</sup>A-Net and S<sup>2</sup>A-Net is:
* CS<sup>2</sup>A-Net head aligns features first by an Alignment Convolution Layer (ACL), and produce _cls_score_ and _bbox_pred_ later.
While the FAM in S<sup>2</sup>A-Net does clssification and regression first, then produces aligned features.

* CS<sup>2</sup>A-Net aligns features in each stage, while feature alignment in S<sup>2</sup>A-Net is only appeared at the end of FAM. 
So even the one stage version of CS<sup>2</sup>A-Net, _.i.e_, CS<sup>2</sup>A-Net-1s can produce aligned features.
