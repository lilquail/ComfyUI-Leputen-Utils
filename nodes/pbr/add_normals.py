"""
Add Normals node - blends two normal maps together.
"""

import torch

from ...core import LEPUTEN_UTILS_CATEGORY, image_to_normal, normal_to_image


class AddNormals:
    """
    Combines two normal maps (a base and a detail map) using various blending algorithms.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_normal": (
                    "IMAGE",
                    {"tooltip": "The primary normal map, typically representing larger surface details."},
                ),
                "detail_normal": (
                    "IMAGE",
                    {
                        "tooltip": "The secondary normal map containing finer details to be layered onto the base normal map."
                    },
                ),
                "detail_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 4.0,
                        "step": 0.01,
                        "tooltip": "Controls the intensity of the detail normal map's effect on the base normal map.",
                    },
                ),
                "method": (
                    ["UDN", "Whiteout", "Partial Derivatives", "Reoriented Normal Mapping"],
                    {
                        "default": "UDN",
                        "tooltip": "UDN: Fast, good for overlays. Whiteout: Industry standard, preserves detail. Partial Derivatives: Mathematical blend. Reoriented: Most accurate, slowest.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_normals"
    CATEGORY = f"{LEPUTEN_UTILS_CATEGORY}/NormalMap"
    DESCRIPTION = """Blends two normal maps (base + detail) together.

• UDN: Simple and fast. Adds XY components, keeps base Z. Good for subtle overlays.
• Whiteout: Industry standard (Unity, Unreal). Preserves detail in both maps. Best for most cases.
• Partial Derivatives: Mathematical blend using surface gradients. Accurate but can flatten.
• Reoriented Normal Mapping: Most physically accurate. Slower but handles steep normals best."""

    def add_normals(
        self, base_normal: torch.Tensor, detail_normal: torch.Tensor, detail_strength: float, method: str
    ) -> tuple[torch.Tensor]:
        n1 = image_to_normal(base_normal)
        n2 = image_to_normal(detail_normal)

        flat_normal = torch.tensor([0.0, 0.0, 1.0]).view(1, 1, 1, 3).to(n2.device)
        n2 = flat_normal + (n2 - flat_normal) * detail_strength

        # Apply detail normal maps using chosen method
        if method == "UDN":
            # UDN (Unreal Developer Network): Sums XY channels, keeps base Z.
            # Good for performance, but can look flat on steep normals.
            blended_n = torch.cat([n1[..., 0:2] + n2[..., 0:2], n1[..., 2:3]], dim=-1)
        elif method == "Whiteout":
            # Whiteout: Standard industry blend (Unity/Unreal).
            # Multiples Z channels for better volume preservation.
            blended_n = torch.cat([n1[..., 0:2] + n2[..., 0:2], n1[..., 2:3] * n2[..., 2:3]], dim=-1)
        elif method == "Partial Derivatives":
            # Partial Derivatives: Mathematical blend that treats normals as surface slopes.
            blended_n = torch.cat(
                [n1[..., 0:2] * n2[..., 2:3] + n2[..., 0:2] * n1[..., 2:3], n1[..., 2:3] * n2[..., 2:3]], dim=-1
            )
        elif method == "Reoriented Normal Mapping":
            # RNM (Stephen Hill): Most physically accurate blending method.
            # Effectively rotates the detail normal map to follow the base map's surface curve.
            t_prime = torch.cat([n1[..., 0:2], n1[..., 2:3] + 1.0], dim=-1)
            u_prime = torch.cat([-n2[..., 0:2], n2[..., 2:3]], dim=-1)
            t_dot_u = torch.sum(t_prime * u_prime, dim=-1, keepdim=True)
            t_prime_z = t_prime[..., 2:3]
            t_prime_z[t_prime_z == 0] = 1e-6
            blended_n = t_prime * (t_dot_u / t_prime_z) - u_prime

        final_norm = torch.nn.functional.normalize(blended_n, p=2, dim=-1)
        output_image = normal_to_image(final_norm)

        return (output_image,)
