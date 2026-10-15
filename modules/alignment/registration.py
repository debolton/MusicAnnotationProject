# modules/alignment/registration.py

import cv2
import numpy as np


class RegistrationEngine:

    def __init__(self, registration_config):

        self.config = registration_config

    # --------------------------------------------------
    # Alignment
    # --------------------------------------------------

    def align_pair(
        self,
        clean_path,
        annotated_path,
    ):

        clean = cv2.imread(
            str(clean_path),
            cv2.IMREAD_GRAYSCALE,
        )

        annotated = cv2.imread(
            str(annotated_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if clean is None:

            raise ValueError(
                f"Unreadable clean image: "
                f"{clean_path}"
            )

        if annotated is None:

            raise ValueError(
                f"Unreadable annotated image: "
                f"{annotated_path}"
            )

        #
        # Scale images for feature extraction
        #

        clean_scale = min(

            1.0,

            self.config[
                "max_feature_dimension"
            ]
            /
            max(clean.shape)

        )

        annotated_scale = min(

            1.0,

            self.config[
                "max_feature_dimension"
            ]
            /
            max(annotated.shape)

        )

        clean_small = cv2.resize(

            clean,

            None,

            fx=clean_scale,

            fy=clean_scale,

            interpolation=
            cv2.INTER_AREA,

        )

        annotated_small = cv2.resize(

            annotated,

            None,

            fx=annotated_scale,

            fy=annotated_scale,

            interpolation=
            cv2.INTER_AREA,

        )

        #
        # CLAHE preprocessing
        #

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )

        clean_clahe = clahe.apply(
            clean_small
        )

        annotated_clahe = (
            clahe.apply(
                annotated_small
            )
        )

        #
        # Feature extraction
        #

        sift = cv2.SIFT_create(

            nfeatures=self.config[
                "sift_features"
            ],

            contrastThreshold=0.015,

            edgeThreshold=15,

        )

        (
            annotated_keypoints,
            annotated_desc,
        ) = sift.detectAndCompute(
            annotated_clahe,
            None,
        )

        (
            clean_keypoints,
            clean_desc,
        ) = sift.detectAndCompute(
            clean_clahe,
            None,
        )

        if annotated_desc is None:

            raise ValueError(
                "Annotated image "
                "contains no SIFT "
                "descriptors"
            )

        if clean_desc is None:

            raise ValueError(
                "Clean image "
                "contains no SIFT "
                "descriptors"
            )

        #
        # FLANN matching
        #

        matcher = cv2.FlannBasedMatcher(

            dict(
                algorithm=1,
                trees=5,
            ),

            dict(
                checks=120,
            ),
        )

        matches = matcher.knnMatch(

            annotated_desc,
            clean_desc,
            k=2,
        )

        good_matches = []

        for pair in matches:

            if len(pair) != 2:
                continue

            first, second = pair

            if (

                first.distance

                <

                self.config[
                    "ratio_test"
                ]
                *
                second.distance

            ):

                good_matches.append(
                    first
                )

        if len(good_matches) < 20:

            raise ValueError(

                f"Only "
                f"{len(good_matches)} "
                f"good matches found"

            )

        #
        # Homography
        #

        src_points = np.float32(

            [

                annotated_keypoints[
                    m.queryIdx
                ].pt

                for m

                in good_matches

            ]

        )

        dst_points = np.float32(

            [

                clean_keypoints[
                    m.trainIdx
                ].pt

                for m

                in good_matches

            ]

        )

        src_points = (
            src_points
            / annotated_scale
        )

        dst_points = (
            dst_points
            / clean_scale
        )

        (
            homography,
            inliers,
        ) = cv2.findHomography(

            src_points,

            dst_points,

            cv2.RANSAC,

            self.config[
                "ransac_threshold"
            ],

            maxIters=15000,

            confidence=0.999,

        )

        if homography is None:

            raise ValueError(
                "Homography estimation "
                "failed"
            )

        #
        # Warp annotated page into
        # clean-page coordinate system
        #

        aligned = (
            cv2.warpPerspective(

                annotated,

                homography,

                (
                    clean.shape[1],
                    clean.shape[0],
                ),

                flags=
                cv2.INTER_CUBIC,

                borderValue=255,

            )
        )

        diagnostics = {

            "matches":
                len(
                    good_matches
                ),

            "inliers":
                int(
                    inliers.sum()
                ),

            "inlier_percentage":

                round(

                    100

                    * int(
                        inliers.sum()
                    )

                    /

                    len(
                        good_matches
                    ),

                    2,

                ),

            "clean_shape":
                list(
                    clean.shape
                ),

            "annotated_shape":
                list(
                    annotated.shape
                ),
        }

        return (

            clean,

            aligned,

            homography,

            diagnostics,

        )

    # --------------------------------------------------
    # Review image
    # --------------------------------------------------

    @staticmethod
    def registration_preview(
        clean,
        aligned,
    ):

        preview = np.zeros(

            (
                clean.shape[0],
                clean.shape[1],
                3,
            ),

            np.uint8,

        )

        #
        # Green = clean
        # Red = aligned
        #

        preview[:, :, 1] = clean

        preview[:, :, 2] = aligned

        return preview
