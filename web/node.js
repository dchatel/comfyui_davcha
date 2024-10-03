import { app } from "/scripts/app.js"
import { api } from "/scripts/api.js"

const loaddata = app.loadGraphData;

app.registerExtension({
    name: "comfyui_davcha.DavchaLoadVideo",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("davcha")) {
            return;
        }

        if (nodeData.name == "DavchaLoadVideo") {
            nodeType.prototype.onDragOver = function (e) {
                if (e.dataTransfer && e.dataTransfer.items) {
                    return [...e.dataTransfer.items].some(f => f.kind === "file" && (f.type.startsWith("image/") || f.type.startsWith("video/")));
                }
            };


            nodeType.prototype.onDragDrop = async function (e) {
                const file = e.dataTransfer.files[0];

                const body = new FormData();
                body.append("image", file);
                // if (pasted) body.append("subfolder", "pasted");
                const resp = await api.fetchApi("/upload/image", {
                    method: "POST",
                    body
                });
                const data = await resp.json();

                this.widgets.find(w => w.name == 'path').value = data.name;
            };
        }

        const getOptions = function(node) {
            const width = node.widgets.find(w => w.name == "width").value;
            const height = node.widgets.find(w => w.name == "height").value;
            const x = Array.from({ length: (width + 1) / 8 + 1 }, (_, i) => i * 8);
            const y = Array.from({ length: (height + 1) / 8 + 1 }, (_, i) => i * 8);

            const lastX = x[x.length - 1];
            const lastY = y[y.length - 1];
            const ratio = lastX / lastY;

            const options = [];

            for (let i = 0; i < x.length; i++) {
                for (let j = 0; j < y.length; j++) {
                    if (x[i] / y[j] === ratio && Math.round(lastX / x[i] * 100) / 100 === lastX / x[i]) {
                        options.push(`${x[i]}x${y[j]}: ${lastX / x[i]}`);
                    }
                }
            }
            return options;
        }

        const update = function (node, option_value = null) {
            const width = node.widgets.find(w => w.name == "width").value;
            const height = node.widgets.find(w => w.name == "height").value;
            const x = Array.from({ length: (width + 1) / 8 + 1 }, (_, i) => i * 8);
            const y = Array.from({ length: (height + 1) / 8 + 1 }, (_, i) => i * 8);

            const lastX = x[x.length - 1];
            const lastY = y[y.length - 1];
            const ratio = lastX / lastY;

            const options = [];

            for (let i = 0; i < x.length; i++) {
                for (let j = 0; j < y.length; j++) {
                    if (x[i] / y[j] === ratio && Math.round(lastX / x[i] * 100) / 100 === lastX / x[i]) {
                        options.push(`${x[i]}x${y[j]}: ${lastX / x[i]}`);
                    }
                }
            }

            const option = node.widgets.find(w => w.name == 'option')
            option.options = { values: options };
            if (option_value !== null && options.indexOf(option_value) != -1) {
                option.value = option_value;
            } else {
                option.value = options[0];
            }
        }

        if (nodeData.name == 'DavchaEmptyLatentImage') {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                this.widgets.find(w => w.name == 'width').callback = () => update(this);
                this.widgets.find(w => w.name == 'height').callback = () => update(this);
                update(this);
            };
            
            nodeType.prototype.onGraphConfigured = function () {
                const option_value = this.widgets.find(w => w.name == 'option').value;
                update(this, option_value);
            };

            nodeType.prototype.refreshComboInNode = function(defs){
                const option_value = this.widgets.find(w => w.name == 'option').value;
                const options = getOptions(this);
                defs['DavchaEmptyLatentImage']['input']['required']['option'] = [options, {}];
            };
        }
    },

    async setup() {
        const head = document.getElementsByTagName('HEAD')[0];
        const link = document.createElement('link');
        link.rel = 'stylesheet';
        link.type = 'text/css';
        link.href = 'extensions/comfyui_davcha/style.css';
        head.appendChild(link);
    },

})