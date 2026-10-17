#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { join } from "node:path";

const require = createRequire(import.meta.url);
const platforms = {
  "darwin-arm64": "@openviking/compile-darwin-arm64",
  "darwin-x64": "@openviking/compile-darwin-x64",
  "linux-arm64": "@openviking/compile-linux-arm64",
  "linux-x64": "@openviking/compile-linux-x64",
  "win32-x64": "@openviking/compile-win32-x64"
};

const key = `${process.platform}-${process.arch}`;
const packageName = platforms[key];
if (!packageName) {
  console.error(`Unsupported platform: ${key}`);
  process.exit(1);
}

let packageJson;
try {
  packageJson = require.resolve(`${packageName}/package.json`);
} catch {
  console.error(
    `Native package ${packageName} is missing. Reinstall without --omit=optional.`
  );
  process.exit(1);
}

const extension = process.platform === "win32" ? ".exe" : "";
const binary = join(packageJson, "..", "bin", `ov${extension}`);

try {
  execFileSync(binary, process.argv.slice(2), { stdio: "inherit" });
} catch (error) {
  process.exit(error.status ?? 1);
}
