import { app } from "../../scripts/app.js";

/**
 * DDS Preview Widget Extension
 * 
 * Adds an image preview to DDSLoader nodes using the same pattern
 * as the native LoadImage node (node.imgs array).
 */
app.registerExtension({
    name: "Leputen.Utils.DDSPreview",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "DDSLoader") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);

            const imageWidget = this.widgets?.find(w => w.name === "image");
            if (!imageWidget) {
                return result;
            }

            const updatePreview = (filename) => {
                if (!filename) {
                    this.imgs = null;
                    this.setSizeForImage?.();
                    return;
                }

                const img = new Image();
                img.src = `/leputen/dds_preview?filename=${encodeURIComponent(filename)}`;

                img.onload = () => {
                    this.imgs = [img];
                    this.setSizeForImage?.();
                    app.graph.setDirtyCanvas(true, true);
                };

                img.onerror = () => {
                    this.imgs = null;
                    console.warn(`[Leputen] Failed to load DDS preview: ${filename}`);
                };
            };

            const originalCallback = imageWidget.callback;
            imageWidget.callback = (value) => {
                if (originalCallback) {
                    originalCallback.call(this, value);
                }
                updatePreview(value);
            };

            if (imageWidget.value) {
                // Delay initial load to ensure node is fully initialized
                setTimeout(() => updatePreview(imageWidget.value), 100);
            }

            return result;
        };
    },
});
