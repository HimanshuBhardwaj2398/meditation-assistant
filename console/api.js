export class ApiError extends Error {
  constructor(status) {
    super("HTTP " + status);
    this.status = status;
  }
}

async function getJson(path, signal) {
  const response = await fetch(path, { signal });
  if (!response.ok) {
    throw new ApiError(response.status);
  }
  return response.json();
}

export function listDocuments({ nikaya, limit, offset, signal }) {
  const params = new URLSearchParams({ limit, offset });
  if (nikaya) {
    params.set("nikaya", nikaya);
  }
  return getJson("/v1/documents?" + params, signal);
}

export function getChunks(documentId, signal) {
  return getJson("/v1/documents/" + documentId + "/chunks", signal);
}

export function getDocument(documentId, signal) {
  return getJson("/v1/documents/" + documentId, signal);
}
