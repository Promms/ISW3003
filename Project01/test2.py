from data.pascal_voc import get_loader

# 1. 실행 로직을 main 함수나 if __name__ == '__main__': 안에 넣습니다.
if __name__ == '__main__':
    train_loader = get_loader("./data", image_set="train", batch_size=2, download=True)
    val_loader   = get_loader("./data", image_set="val",   batch_size=2)

    # train_loader 테스트
    image, mask = next(iter(train_loader))
    print(f"Train Image: {image.shape}")    # (2, 3, 320, 320)
    print(f"Train Mask: {mask.shape}")      # (2, 320, 320)
    print(f"Unique Labels: {mask.unique()}") # tensor([0, 1, ..., 255])

    # val_loader 테스트
    image, mask = next(iter(val_loader))
    print(f"Val Image: {image.shape}")      # (2, 3, 480, 640)
    print(f"Val Mask: {mask.shape}")        # (2, 480, 640)