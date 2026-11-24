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

let text = await app.vault.read(activeFile);
let addedSections = 0;
let addedPlots = 0;

// 1. Gather current .ipynb, .py, and numbered subdirectories
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

// 2. Append any newly created sections at the bottom
for (const sec of sections) {
    const header = `\n\n## \`${sec}\``;
    if (!text.includes(`## \`${sec}\``)) {
        text += `${header}\n- blah blah \n- **Plots:**\n`;
        addedSections++;
    }
}

// 3. Scan 'plots' subfolder and insert missing plots in numeric order
const plotsFolder = children.find(c => Array.isArray(c.children) && c.name === "plots");

if (plotsFolder && plotsFolder.children) {
    const plotNames = plotsFolder.children
        .filter(f => !Array.isArray(f.children) && f.extension === "png")
        .map(f => f.name)
        .sort(naturalSort);

    for (const sec of sections) {
        const prefix = getLeadingPrefix(sec);
        if (!prefix) continue;

        const matchingPlots = plotNames.filter(plot => getLeadingPrefix(plot) === prefix);
        if (matchingPlots.length === 0) continue;

        const secHeader = `## \`${sec}\``;
        const secPos = text.indexOf(secHeader);
        if (secPos === -1) continue;

        for (const plot of matchingPlots) {
            if (text.includes(`./plots/${plot}`) || text.includes(`![[${plot}`)) {
                continue;
            }

            const newPlotNum = getPlotNumber(plot);
            const plotBlock = `    - **${plot}**![${plot}](./plots/${plot})\n        - blah blah\n`;

            // Scope search strictly to this section
            const nextSecPos = text.indexOf("\n## `", secPos + secHeader.length);
            const sectionEnd = nextSecPos === -1 ? text.length : nextSecPos;
            const sectionSlice = text.slice(secPos, sectionEnd);

            const lines = sectionSlice.match(/[^\n]*\n|[^\n]+/g) || [];
            let inserted = false;
            let offset = secPos;

            for (const line of lines) {
                const stripped = line.trimStart();
                if (stripped.startsWith("- **") && stripped.indexOf("**", 4) !== -1) {
                    const endBold = stripped.indexOf("**", 4);
                    const plotTitle = stripped.slice(4, endBold);
                    const existingNum = getPlotNumber(plotTitle);

                    if (newPlotNum < existingNum) {
                        text = text.slice(0, offset) + plotBlock + text.slice(offset);
                        inserted = true;
                        break;
                    }
                }
                offset += line.length;
            }

            // Fallback: append to the end of the section if it has the highest plot number
            if (!inserted) {
                const plotsTag = "- **Plots:**";
                const plotsTagPos = text.indexOf(plotsTag, secPos);
                if (plotsTagPos !== -1) {
                    const insertPos = plotsTagPos + plotsTag.length;
                    const nextSectionPos = text.indexOf("\n\n## `", insertPos);
                    if (nextSectionPos === -1) {
                        if (!text.endsWith("\n")) text += "\n";
                        text += plotBlock;
                    } else {
                        text = text.slice(0, nextSectionPos) + "\n" + plotBlock.trimEnd() + text.slice(nextSectionPos);
                    }
                }
            }
            addedPlots++;
        }
    }
}

// 4. Save changes or notify if nothing changed
if (addedSections === 0 && addedPlots === 0) {
    new Notice("Already up to date—no new sections or plots found.");
} else {
    await app.vault.modify(activeFile, text);
    new Notice(`Synced: +${addedSections} section(s), +${addedPlots} plot(s).`);
}
%>