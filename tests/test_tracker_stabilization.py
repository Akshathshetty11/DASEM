from src.tracker import VehicleTracker


def _detection(label, confidence=0.9, x=100, y=100, width=80, height=45):
    return {
        "class": label,
        "confidence": confidence,
        "bbox": {"x1": x, "y1": y, "x2": x + width, "y2": y + height},
    }


def test_track_class_is_not_changed_by_single_noisy_observation():
    tracker = VehicleTracker(min_displacement_threshold=0)
    for label, confidence in [("car", .93), ("car", .91), ("car", .89), ("car", .92), ("truck", .52)]:
        detections = [_detection(label, confidence)]
        tracker.update(detections)

    assert detections[0]["track_id"] == 1
    assert tracker.get_stabilized_class(1) == "car"


def test_completed_track_still_contributes_to_unique_counting():
    tracker = VehicleTracker(max_disappeared=1, min_displacement_threshold=0)
    for x in (100, 105, 110, 115):
        tracker.update([_detection("bus", .9, x=x, width=120, height=60)])

    tracker.update([])
    tracker.update([])

    assert tracker.get_all_stabilized_classes(moving_only=False) == {1: "bus"}
    assert tracker.get_all_stabilized_classes(moving_only=True) == {1: "bus"}


def test_crossing_tracks_keep_their_ids():
    tracker = VehicleTracker(max_distance_threshold=80, min_displacement_threshold=0)
    first = [_detection("car", x=40), _detection("bus", x=180, width=120, height=60)]
    tracker.update(first)
    assert [item["track_id"] for item in first] == [1, 2]

    second = [_detection("car", x=95), _detection("bus", x=125, width=120, height=60)]
    tracker.update(second)
    third = [_detection("car", x=150), _detection("bus", x=70, width=120, height=60)]
    tracker.update(third)

    assert [item["track_id"] for item in third] == [1, 2]
