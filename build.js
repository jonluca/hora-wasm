import shell from "shelljs";

if (!shell.which("wasm-pack")) {
  console.error("Run npm install to install all dependencies first");
}

// Clean up any existing built content:
shell.rm("-rf", "dist");
shell.rm("-rf", "pkg.*");
shell.mkdir("dist");

// Create the web output
shell.rm("-rf", "pkg");
shell.exec("wasm-pack build --target web --release");
shell.rm("pkg/{LICENSE,package.json,README.md,.gitignore}");
