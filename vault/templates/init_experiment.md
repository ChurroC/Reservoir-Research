<%*
const naturalSort = (a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' });
const getLeadingPrefix = (name) => (name.match(/^\d+/) || [""])[0];
const getPlotNumber = (filename) => {
    const cleaned = filename.split("_")[0].replace(/[^0-9.]/g, "");
    return parseFloat(cleaned) || 0.0;
};

const activeFile = tp.config.target_file;
const expFolder = activeFile.parent;

if (!expFolder || expFolder.isRoot()) {
    new Notice("Error: Note must be inside an experiment folder.");
    return;
}

// 1. Gather .ipynb, .py, and numbered subdirectories (excluding 'plots')
const children = expFolder.children || [];
const fileNames = [];

for (const child of children) {
    if (Array.isArray(child.children)) {
        if (child.name !== "plots" && /^\d/.test(child.name)) fileNames.push(child.name);
    } else if (child.extension === "ipynb" || child.extension === "py") {
        fileNames.push(child.name);
    }
}

const sections = fileNames.sort(naturalSort);

// 2. Gather and sort all PNGs in 'plots' subfolder
const plotsFolder = children.find(c => Array.isArray(c.children) && c.name === "plots");
const plotNames = (plotsFolder && plotsFolder.children)
    ? plotsFolder.children
        .filter(f => !Array.isArray(f.children) && f.extension === "png")
        .map(f => f.name)
        .sort((a, b) => getPlotNumber(a) - getPlotNumber(b) || naturalSort(a, b))
    : [];

// 3. Build the initial note content
let output = `\nblah blah`;

for (const sec of sections) {
    output += `\n\n## \`${sec}\`\n- blah blah \n- **Plots:**\n`;
    
    const prefix = getLeadingPrefix(sec);
    if (!prefix) continue;

    const matchingPlots = plotNames.filter(plot => getLeadingPrefix(plot) === prefix);
    for (const plot of matchingPlots) {
        output += `    - **${plot}**![${plot}](./plots/${plot})\n`;
        output += `        - blah blah\n`;
    }
}

// 4. Write directly to the note
tR += output;
new Notice(`Initialized Experiment Note: ${expFolder.name}`);
%>
