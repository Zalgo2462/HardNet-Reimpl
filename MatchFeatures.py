import cv2
import numpy as np
import torch

from HardNet import HardNet
from HardNetModule import HardNetModule


def main():
    img1_path = "/home/logan/Documents/classes/computer-vision/lembke_project_2/images/bikes/img1.ppm"
    img2_path = "/home/logan/Documents/classes/computer-vision/lembke_project_2/images/bikes/img4.ppm"

    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    key_points1, orb_desc1 = orb.detectAndCompute(img1, None)
    key_points2, orb_desc2 = orb.detectAndCompute(img2, None)

    img1_patches = []
    for point in map(lambda kp: kp.pt, key_points1):
        x_min = int(point[0] - 16)
        x_max = int(point[0] + 16)
        y_min = int(point[1] - 16)
        y_max = int(point[1] + 16)

        img1_patches.append(img1[y_min:y_max, x_min:x_max])

    img2_patches = []
    for point in map(lambda kp: kp.pt, key_points2):
        x_min = int(point[0] - 16)
        x_max = int(point[0] + 16)
        y_min = int(point[1] - 16)
        y_max = int(point[1] + 16)

        img2_patches.append(img2[y_min:y_max, x_min:x_max])

    img1_patch_tensor = torch.FloatTensor(np.expand_dims(np.stack(img1_patches), axis=1))
    img2_patch_tensor = torch.FloatTensor(np.expand_dims(np.stack(img1_patches), axis=1))

    hardNet = HardNet(HardNetModule(), "")
    hardNet.load_checkpoint("./data/models/linux13_run_1/checkpoint_9.pth")
    hn_desc1 = hardNet.create_descriptors(img1_patch_tensor).numpy()
    hn_desc2 = hardNet.create_descriptors(img2_patch_tensor).numpy()

    orb_matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
    hn_matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)

    orb_matches = orb_matcher.match(orb_desc1, orb_desc2)
    hn_matches = hn_matcher.match(hn_desc1, hn_desc2)

    orb_matches = sorted(orb_matches, key=lambda x: x.distance)[:100]
    hn_matches = sorted(hn_matches, key=lambda x: x.distance)[:100]


    orb_img = cv2.drawMatches(img1, key_points1, img2, key_points2, orb_matches, None, flags=2)

    hn_img = cv2.drawMatches(img1, key_points1, img2, key_points2, hn_matches, None, flags=2)

    cv2.imshow("ORB Matches", orb_img)
    cv2.imshow("HardNet Matches", hn_img)
    cv2.waitKey()


if __name__ == "__main__":
    main()
