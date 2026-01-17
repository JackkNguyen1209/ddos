import express, { type Express, Request, Response, NextFunction } from "express";
import fs from "fs";
import path from "path";

export function serveStatic(app: Express) {
  const distPath = path.resolve(__dirname, "public");
  if (!fs.existsSync(distPath)) {
    throw new Error(
      `Could not find the build directory: ${distPath}, make sure to build the client first`,
    );
  }

  app.use(express.static(distPath));

  // SPA fallback middleware - serve index.html for all unmatched GET requests
  // This must be after static files and API routes
  app.use((_req: Request, res: Response, _next: NextFunction) => {
    res.sendFile(path.resolve(distPath, "index.html"));
  });
}
