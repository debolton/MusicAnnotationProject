from modules.alignment.auto_mask import (
    AutoMaskGenerator
)

generator = AutoMaskGenerator(
    CONFIG["auto_mask"]
)

strength, mask = (
    generator.generate(
        clean,
        aligned
    )
)

stats = (
    generator.mask_statistics(
        mask
    )
)
