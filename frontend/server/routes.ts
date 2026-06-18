import type { Express } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";

export async function registerRoutes(app: Express): Promise<Server> {
  // Proxy all requests starting with /api to the Python FastAPI backend
  app.all("/api/*", async (req, res) => {
    const backendUrl = `http://127.0.0.1:8000${req.originalUrl}`;
    try {
      const options: RequestInit = {
        method: req.method,
      };
      const headers: Record<string, string> = {};

      // Copy incoming headers (excluding host)
      for (const [key, value] of Object.entries(req.headers)) {
        if (key.toLowerCase() !== "host" && typeof value === "string") {
          headers[key] = value;
        }
      }

      // Handle body if present
      if (["POST", "PUT", "PATCH", "DELETE"].includes(req.method) && req.body) {
        options.body = JSON.stringify(req.body);
        headers["Content-Type"] = "application/json";
      }

      options.headers = headers;

      const backendResponse = await fetch(backendUrl, options);
      const data = await backendResponse.text();

      res.status(backendResponse.status);
      
      // Copy headers from backend response (excluding transport encodings)
      backendResponse.headers.forEach((value, key) => {
        const lowerKey = key.toLowerCase();
        if (!["transfer-encoding", "content-encoding", "content-length"].includes(lowerKey)) {
          res.setHeader(key, value);
        }
      });
      
      res.send(data);
    } catch (error: any) {
      console.error(`Proxy error to ${backendUrl}:`, error);
      res.status(502).json({ error: "Failed to connect to backend service" });
    }
  });

  const httpServer = createServer(app);

  return httpServer;
}
