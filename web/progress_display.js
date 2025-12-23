import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

/**
 * Progress Display Widget Extension
 * 
 * Displays inline progress/status text on nodes using a read-only text widget.
 * Uses the same pattern as pythongosssss/ComfyUI-Custom-Scripts ShowText node.
 * 
 * Supported nodes:
 * - ImageIterator, DDSIterator: Shows batch progress (e.g., "Batch 1/5")
 * - LoadCubemapFaces: Shows found faces info
 */

const SUPPORTED_NODES = ["ImageIterator", "DDSIterator", "LoadCubemapFaces"];

// Keywords to detect saved progress text for workflow restore
const RESTORE_KEYWORDS = ["Batch", "Found"];

app.registerExtension({
    name: "Leputen.Utils.ProgressDisplay",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!SUPPORTED_NODES.includes(nodeData.name)) {
            return;
        }

        function updateProgressWidget(text) {
            // Remove existing progress widget if any
            if (this._progressWidget) {
                const idx = this.widgets?.indexOf(this._progressWidget);
                if (idx > -1) {
                    this._progressWidget.onRemove?.();
                    this.widgets.splice(idx, 1);
                }
            }

            // Create new widget with the text
            const w = ComfyWidgets["STRING"](
                this,
                "progress_display",
                ["STRING", { multiline: true }],
                app
            ).widget;

            w.inputEl.readOnly = true;
            w.inputEl.style.opacity = "0.7";
            w.value = text;

            this._progressWidget = w;

            // Resize node to fit content
            requestAnimationFrame(() => {
                const sz = this.computeSize();
                if (sz[0] < this.size[0]) {
                    sz[0] = this.size[0];
                }
                if (sz[1] < this.size[1]) {
                    sz[1] = this.size[1];
                }
                this.onResize?.(sz);
                app.graph.setDirtyCanvas(true, false);
            });
        }

        // Handle executed event to update progress text
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (message?.progress_text?.[0]) {
                updateProgressWidget.call(this, message.progress_text[0]);
            }
        };

        // Restore widget on configure (when loading workflow)
        const VALUES = Symbol();
        const configure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            this[VALUES] = arguments[0]?.widgets_values;
            return configure?.apply(this, arguments);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            const widgets_values = this[VALUES];

            // Look for saved progress text by checking for known keywords
            if (widgets_values?.length) {
                const lastValue = widgets_values[widgets_values.length - 1];
                if (lastValue && typeof lastValue === "string" &&
                    RESTORE_KEYWORDS.some(keyword => lastValue.includes(keyword))) {
                    requestAnimationFrame(() => {
                        updateProgressWidget.call(this, lastValue);
                    });
                }
            }
        };
    },
});
