#!/usr/bin/env bun

const fs = require("fs");
const path = require("path");

function usage() {
  console.error(
    [
      "Usage:",
      "  bun scripts/list-ts-dependencies.js <path-to-file.ts> <parent-ts-package>",
      "",
      "Examples:",
      "  bun scripts/list-ts-dependencies.js ../packages/app/src/foo.ts ../packages/app",
      "  bun scripts/list-ts-dependencies.js src/foo.ts @scope/app",
    ].join("\n"),
  );
}

function loadParser() {
  try {
    const Parser = require("tree-sitter");
    const TypeScript = require("tree-sitter-typescript").typescript;
    const parser = new Parser();
    parser.setLanguage(TypeScript);
    return parser;
  } catch (error) {
    console.error("Unable to load Tree-sitter dependencies.");
    console.error("Run `bun install` from `experimental-editor` first.");
    console.error(`Original error: ${error.message}`);
    process.exit(1);
  }
}

function readJsonIfExists(filePath) {
  if (!fs.existsSync(filePath)) {
    return null;
  }

  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function nearestPackageRoot(startPath) {
  let current = fs.statSync(startPath).isDirectory() ? startPath : path.dirname(startPath);
  const root = path.parse(current).root;

  while (current !== root) {
    if (fs.existsSync(path.join(current, "package.json"))) {
      return current;
    }
    current = path.dirname(current);
  }

  return process.cwd();
}

function normalizeParentPackage(parentArg, sourcePath) {
  const absoluteCandidate = path.resolve(parentArg);

  if (fs.existsSync(absoluteCandidate) && fs.statSync(absoluteCandidate).isDirectory()) {
    const packageJson = readJsonIfExists(path.join(absoluteCandidate, "package.json"));
    return {
      root: absoluteCandidate,
      name: packageJson && packageJson.name ? packageJson.name : path.basename(absoluteCandidate),
    };
  }

  return {
    root: nearestPackageRoot(sourcePath),
    name: parentArg,
  };
}

function stripQuotes(value) {
  return value.replace(/^["'`]|["'`]$/g, "");
}

function stripTypeOnlyPrefix(value) {
  return value.replace(/^type\s+/, "").trim();
}

function withoutExtension(value) {
  return value.replace(/\.(tsx?|jsx?)$/, "");
}

function sourceLabel(moduleSpecifier, importerPath, parentPackage) {
  if (!moduleSpecifier.startsWith(".")) {
    return moduleSpecifier;
  }

  const resolved = path.resolve(path.dirname(importerPath), moduleSpecifier);
  const relativeToRoot = path.relative(parentPackage.root, resolved);

  if (relativeToRoot.startsWith("..") || path.isAbsolute(relativeToRoot)) {
    return moduleSpecifier;
  }

  const packageRelative = withoutExtension(relativeToRoot).split(path.sep).join("/");
  return `${parentPackage.name}/${packageRelative}`.replace(/\/index$/, "");
}

function textFor(source, node) {
  return source.slice(node.startIndex, node.endIndex);
}

function walk(node, visit) {
  visit(node);
  for (let index = 0; index < node.namedChildCount; index += 1) {
    walk(node.namedChild(index), visit);
  }
}

function findNamedChildren(node, type) {
  const matches = [];
  for (let index = 0; index < node.namedChildCount; index += 1) {
    const child = node.namedChild(index);
    if (child.type === type) {
      matches.push(child);
    }
  }
  return matches;
}

function importSourceFromText(importText) {
  const fromMatch = importText.match(/\bfrom\s+(["'`][^"'`]+["'`])/);
  if (fromMatch) {
    return stripQuotes(fromMatch[1]);
  }

  const sideEffectMatch = importText.match(/^import\s+(["'`][^"'`]+["'`])\s*;?$/);
  return sideEffectMatch ? stripQuotes(sideEffectMatch[1]) : null;
}

function splitTopLevel(value) {
  const parts = [];
  let start = 0;
  let depth = 0;

  for (let index = 0; index < value.length; index += 1) {
    const char = value[index];
    if (char === "{" || char === "(" || char === "[") depth += 1;
    if (char === "}" || char === ")" || char === "]") depth -= 1;
    if (char === "," && depth === 0) {
      parts.push(value.slice(start, index).trim());
      start = index + 1;
    }
  }

  const last = value.slice(start).trim();
  if (last) {
    parts.push(last);
  }

  return parts;
}

function parseNamedImport(specifier) {
  const cleanSpecifier = stripTypeOnlyPrefix(specifier);
  const aliasMatch = cleanSpecifier.match(/^(.+?)\s+as\s+([A-Za-z_$][\w$]*)$/);

  if (aliasMatch) {
    return {
      importedName: stripTypeOnlyPrefix(aliasMatch[1].trim()),
      localName: aliasMatch[2],
    };
  }

  return {
    importedName: cleanSpecifier,
    localName: cleanSpecifier,
  };
}

function parseImportSymbols(importText, sourcePath, parentPackage, location) {
  const moduleSpecifier = importSourceFromText(importText);
  if (!moduleSpecifier) {
    return [];
  }

  const source = sourceLabel(moduleSpecifier, sourcePath, parentPackage);
  const fromIndex = importText.search(/\bfrom\s+["'`]/);
  const rawImportClause =
    fromIndex === -1
      ? importText.replace(/^import\s+/, "").replace(/["'`][^"'`]+["'`]\s*;?$/, "").trim()
      : importText.slice("import".length, fromIndex).trim();
  const importClause = rawImportClause.replace(/^type\s+/, "").trim();

  if (!importClause) {
    return [
      {
        kind: "side-effect",
        importedName: "*",
        localName: moduleSpecifier,
        source,
        location,
      },
    ];
  }

  const symbols = [];
  const namespaceMatch = importClause.match(/\*\s+as\s+([A-Za-z_$][\w$]*)/);
  if (namespaceMatch) {
    symbols.push({
      kind: "namespace",
      importedName: "*",
      localName: namespaceMatch[1],
      source,
      location,
    });
  }

  const namedMatch = importClause.match(/\{([\s\S]*)\}/);
  if (namedMatch) {
    for (const specifier of splitTopLevel(namedMatch[1])) {
      const namedImport = parseNamedImport(specifier);
      symbols.push({
        kind: "named",
        importedName: namedImport.importedName,
        localName: namedImport.localName,
        source,
        location,
      });
    }
  }

  const defaultPart = importClause.replace(/\{[\s\S]*\}/, "").replace(/\*\s+as\s+[A-Za-z_$][\w$]*/, "");
  const defaultName = splitTopLevel(defaultPart)[0];
  if (defaultName && /^[A-Za-z_$][\w$]*$/.test(defaultName)) {
    symbols.push({
      kind: "default",
      importedName: "default",
      localName: defaultName,
      source,
      location,
    });
  }

  return symbols;
}

function collectImports(tree, source, sourcePath, parentPackage) {
  const imports = [];

  walk(tree.rootNode, (node) => {
    if (node.type !== "import_statement") {
      return;
    }

    imports.push(
      ...parseImportSymbols(textFor(source, node), sourcePath, parentPackage, {
        line: node.startPosition.row + 1,
        column: node.startPosition.column + 1,
      }),
    );
  });

  return imports;
}

function collectTypeNamesFromNode(source, node) {
  const names = new Set();
  const typeText = textFor(source, node);
  const identifierPattern = /\b[A-Z_$][A-Za-z0-9_$]*\b/g;

  for (const match of typeText.matchAll(identifierPattern)) {
    names.add(match[0]);
  }

  return names;
}

function collectConstructorInjectedTypeNames(tree, source) {
  const injectedTypeNames = new Set();

  walk(tree.rootNode, (node) => {
    if (node.type !== "method_definition") {
      return;
    }

    const methodName = node.childForFieldName("name");
    if (!methodName || textFor(source, methodName) !== "constructor") {
      return;
    }

    const parameters = node.childForFieldName("parameters");
    if (!parameters) {
      return;
    }

    walk(parameters, (parameterNode) => {
      if (!parameterNode.type.includes("parameter")) {
        return;
      }

      for (const typeAnnotation of findNamedChildren(parameterNode, "type_annotation")) {
        for (const name of collectTypeNamesFromNode(source, typeAnnotation)) {
          injectedTypeNames.add(name);
        }
      }
    });
  });

  return injectedTypeNames;
}

function dedupeDependencies(dependencies) {
  const seen = new Set();
  const deduped = [];

  for (const dependency of dependencies) {
    const key = `${dependency.localName}\0${dependency.source}`;
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    deduped.push(dependency);
  }

  return deduped.sort((left, right) => {
    const leftKey = `${left.localName} ${left.source}`;
    const rightKey = `${right.localName} ${right.source}`;
    return leftKey.localeCompare(rightKey);
  });
}

function formatDependency(dependency) {
  const importedSuffix =
    dependency.importedName && dependency.importedName !== dependency.localName
      ? ` (${dependency.importedName})`
      : "";

  return `- ${dependency.localName}${importedSuffix} from ${dependency.source}`;
}

function printCategory(title, dependencies) {
  console.log(`${title}:`);

  if (dependencies.length === 0) {
    console.log("- <none>");
    return;
  }

  for (const dependency of dependencies) {
    console.log(formatDependency(dependency));
  }
}

function main() {
  const [, , fileArg, parentArg] = process.argv;

  if (!fileArg || !parentArg) {
    usage();
    process.exit(1);
  }

  const sourcePath = path.resolve(fileArg);
  if (!fs.existsSync(sourcePath)) {
    console.error(`TypeScript file not found: ${sourcePath}`);
    process.exit(1);
  }

  if (!/\.(ts|tsx)$/.test(sourcePath)) {
    console.error(`Expected a .ts or .tsx file: ${sourcePath}`);
    process.exit(1);
  }

  const source = fs.readFileSync(sourcePath, "utf8");
  const parser = loadParser();
  const tree = parser.parse(source);
  const parentPackage = normalizeParentPackage(parentArg, sourcePath);

  const imports = dedupeDependencies(collectImports(tree, source, sourcePath, parentPackage));
  const injectedTypeNames = collectConstructorInjectedTypeNames(tree, source);
  const injected = dedupeDependencies(imports.filter((dependency) => injectedTypeNames.has(dependency.localName)));
  const injectedKeys = new Set(injected.map((dependency) => `${dependency.localName}\0${dependency.source}`));
  const importsOnly = imports.filter((dependency) => !injectedKeys.has(`${dependency.localName}\0${dependency.source}`));

  printCategory("Imports only", importsOnly);
  console.log("");
  printCategory("Inject via constructor", injected);
}

main();
