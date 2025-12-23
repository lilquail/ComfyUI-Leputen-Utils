import { app } from "../../scripts/app.js";

// Z-Stack node - dynamic expanding inputs
// Adds a new image input slot when the current one is connected
// Pattern based on ImpactPack's MakeImageList implementation

app.registerExtension({
    name: "Leputen.ZStack",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "ZStack") {
            return;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
            if (onConnectionsChange) {
                onConnectionsChange.apply(this, arguments);
            }

            // Stack trace safety checks - prevent breaking ComfyUI internal operations
            const stackTrace = new Error().stack;

            // Skip during subgraph operations
            if (stackTrace.includes('convertToSubgraph') || stackTrace.includes('Subgraph.configure')) {
                return;
            }

            // Skip during workflow loading - just ensure inputs are intact
            if (stackTrace.includes('loadGraphData')) {
                return;
            }

            // Skip during paste operations
            if (stackTrace.includes('pasteFromClipboard')) {
                return;
            }

            // Only process if we have valid link info
            if (!link_info) {
                return;
            }

            // Only handle input connections (type === 1), skip output connections (type === 2)
            if (type === 2) {
                return;
            }

            // Get the input that was connected/disconnected
            const inputName = this.inputs[index]?.name;

            // Only process image inputs, not the mode widget
            if (!inputName || !inputName.startsWith("image_")) {
                return;
            }

            // Handle disconnection - remove the input if it's not the only one
            if (!connected && this.inputs.length > 1) {
                // Don't remove during connect operations (which briefly disconnect)
                if (!stackTrace.includes('LGraphNode.prototype.connect') &&
                    !stackTrace.includes('LGraphNode.connect')) {

                    // Count image inputs
                    const imageInputs = this.inputs.filter(i => i.name.startsWith("image_"));
                    if (imageInputs.length > 1) {
                        this.removeInput(index);
                    }
                }
            }

            // Renumber all image inputs sequentially
            let slot_i = 1;
            for (let i = 0; i < this.inputs.length; i++) {
                const input = this.inputs[i];
                if (input.name.startsWith("image_")) {
                    input.name = `image_${slot_i}`;
                    slot_i++;
                }
            }

            // Add new input when connecting to the last available image slot
            if (connected) {
                // Find highest numbered image input
                const imageInputs = this.inputs.filter(i => i.name.startsWith("image_"));
                const lastImageInput = imageInputs[imageInputs.length - 1];

                // If we connected to the last image slot, add a new one
                if (lastImageInput && this.inputs[index].name === lastImageInput.name) {
                    this.addInput(`image_${slot_i}`, "IMAGE", { shape: 7 });
                }
            }
        };
    },
});
