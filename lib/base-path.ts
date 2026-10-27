// next/link and next/image add the base path on their own; plain URLs built by
// hand (model files fetched by TensorFlow.js, for example) go through this.
export const basePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? "").replace(/\/+$/, "")

export function withBasePath(path: string): string {
  return `${basePath}${path.startsWith("/") ? path : `/${path}`}`
}
