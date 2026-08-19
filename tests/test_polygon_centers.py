import numpy as np

from tools.datasets.build_hf_dataset_from_agentic_assets import mask_to_polygons


def test_mask_to_polygons_returns_center_for_each_changed_region():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:6, 3:7] = 1
    mask[12:16, 13:18] = 1

    polygons, centers, areas = mask_to_polygons(mask)

    assert len(polygons) == 2
    assert len(centers) == len(polygons)
    assert len(areas) == len(polygons)
    for cx, cy in centers:
        assert mask[int(cy), int(cx)] == 1


if __name__ == "__main__":
    test_mask_to_polygons_returns_center_for_each_changed_region()
