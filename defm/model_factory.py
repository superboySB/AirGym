from pathlib import Path
from typing import Optional, Tuple, Union

from lib.network.defm_vit import build_defm_vit_s14, load_defm_weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "trained" / "defm_vit_s14.pth"


def create_defm_model(
    model_name: str,
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 224,
):
    aliases = {"defm_vit_s14", "vit_s14"}
    if model_name not in aliases:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: {sorted(aliases)}")

    model = build_defm_vit_s14(img_size=img_size)

    if pretrained:
        ckpt_path = Path(pretrained_path) if pretrained_path else DEFAULT_WEIGHTS
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                f"Download to {DEFAULT_WEIGHTS} or pass pretrained_path explicitly."
            )
        load_defm_weights(model, str(ckpt_path), strict=True)

    return model


def defm_vit_s14(
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 224,
):
    return create_defm_model(
        model_name="defm_vit_s14",
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        img_size=img_size,
    )
