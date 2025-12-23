import { app } from "../../scripts/app.js";

/**
 * Leputen Utils Settings Extension
 * 
 * Registers settings in the ComfyUI settings panel under "Leputen Utils" category.
 * Also handles injecting setting values into nodes that need them.
 */
app.registerExtension({
    name: "Leputen.Utils.Settings",
    settings: [
        {
            id: "LeputenUtils.LogLevel",
            name: "Log Verbosity",
            type: "combo",
            options: ["Debug", "Info", "Warning", "Error"],
            defaultValue: "Info",
            tooltip: "Controls how much logging output is shown in the console.\n\n• Debug: All messages including verbose debug output\n• Info: Standard info messages and above\n• Warning: Warnings and errors only\n• Error: Errors only",
            category: ["Leputen Utils", "General", "Logging"],
            onChange: (newVal) => {
                console.log(`[Leputen Utils] Log level changed to: ${newVal}`);
            },
        },
        {
            id: "LeputenUtils.GigapixelPath",
            name: "Gigapixel CLI Path",
            type: "text",
            defaultValue: "",
            tooltip: "Full path to the Topaz Gigapixel AI executable (gigapixel.exe).\n\nExample: C:\\Program Files\\Topaz Labs LLC\\Topaz Gigapixel AI\\gigapixel.exe\n\nNote: Requires Gigapixel AI Pro for CLI support.",
            category: ["Leputen Utils", "Upscaling", "Gigapixel"],
        },
    ],

    async setup(app) {
        // Hook into queue prompt to inject settings into nodes
        const originalQueuePrompt = app.queuePrompt;
        app.queuePrompt = async function (number, batchCount) {
            const gigapixelPath = app.ui.settings.getSettingValue("LeputenUtils.GigapixelPath", "");

            // Inject Gigapixel path into all GigapixelCLI nodes
            for (const node of app.graph._nodes) {
                if (node.type === "GigapixelCLI") {
                    const hiddenWidget = node.widgets?.find(w => w.name === "gigapixel_path");
                    if (hiddenWidget) {
                        hiddenWidget.value = gigapixelPath;
                    } else {
                        node.properties = node.properties || {};
                        node.properties.gigapixel_path = gigapixelPath;
                    }
                }
            }

            return originalQueuePrompt.apply(this, arguments);
        };
    },
});
