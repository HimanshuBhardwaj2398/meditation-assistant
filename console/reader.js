import { ApiError, getChunks, getDocument, listDocuments } from "./api.js";

const PAGE_SIZE = 20;
const params = new URLSearchParams(window.location.search);
const filter = document.getElementById("nikaya");
const catalogStatus = document.getElementById("catalog-status");
const documentList = document.getElementById("document-list");
const listSkeleton = document.getElementById("list-skeleton");
const retryList = document.getElementById("retry-list");
const previous = document.getElementById("previous");
const next = document.getElementById("next");
const pageNumber = document.getElementById("page-number");
const readerCode = document.getElementById("reader-code");
const readerTitle = document.getElementById("reader-title");
const readerStatus = document.getElementById("reader-status");
const readerSkeleton = document.getElementById("reader-skeleton");
const details = document.getElementById("document-details");
const detailsStatus = document.getElementById("details-status");
const detailsContent = document.getElementById("details-content");
const retryDetails = document.getElementById("retry-details");
const chunkJump = document.getElementById("chunk-jump");
const chunkSelect = document.getElementById("chunk-select");
const refreshChunks = document.getElementById("refresh-chunks");
const chunks = document.getElementById("chunks");
const dateFormatter = new Intl.DateTimeFormat(undefined, { dateStyle: "medium" });

const initialOffset = Number(params.get("offset"));
const state = {
  offset: Number.isSafeInteger(initialOffset) && initialOffset > 0
    ? Math.floor(initialOffset / PAGE_SIZE) * PAGE_SIZE
    : 0,
  selected: null,
  pageRequest: null,
  chunkRequest: null,
  detailRequest: null,
  detailsLoaded: false,
};

filter.value = params.get("nikaya") ?? "";
if (filter.selectedIndex < 0) filter.value = "";

function updateUrl() {
  const url = new URL(window.location.href);
  for (const [key, value] of [
    ["nikaya", filter.value],
    ["offset", state.offset],
    ["document", state.selected?.id],
  ]) {
    if (value) url.searchParams.set(key, value);
    else url.searchParams.delete(key);
  }
  history.replaceState(null, "", url);
}

function setStatus(element, message, error = false) {
  element.textContent = message;
  element.classList.toggle("is-error", error);
}

function textElement(tag, className, value) {
  const element = document.createElement(tag);
  element.className = className;
  element.textContent = value;
  return element;
}

function documentCode(doc) {
  return doc.uid?.replace(/^([a-z]+)(?=\d)/i, "$1 ").toUpperCase() ?? "#" + doc.id;
}

function resetMetadata() {
  state.detailRequest?.abort();
  state.detailRequest = null;
  state.detailsLoaded = false;
  detailsStatus.hidden = true;
  retryDetails.hidden = true;
  detailsContent.replaceChildren();
}

function resetChunks() {
  state.chunkRequest?.abort();
  readerSkeleton.hidden = true;
  chunkJump.hidden = true;
  chunkSelect.replaceChildren(new Option("Choose a chunk", ""));
  chunks.replaceChildren();
}

function clearReader(message) {
  resetChunks();
  resetMetadata();
  state.selected = null;
  readerTitle.textContent = "Reader";
  readerCode.hidden = true;
  details.hidden = true;
  setStatus(readerStatus, message);
  refreshChunks.hidden = true;
}

function renderDocuments(items) {
  const rows = items.map((doc) => {
    const row = document.createElement("li");
    const button = document.createElement("button");
    button.className = "document-button";
    button.type = "button";
    button.dataset.documentId = doc.id;
    button.append(
      textElement("span", "document-code", documentCode(doc)),
      textElement("span", "document-name", doc.title),
    );
    button.addEventListener("click", () => {
      if (state.selected?.id !== doc.id) readDocument(doc);
      if (window.matchMedia("(max-width: 48rem)").matches) {
        readerTitle.focus({ preventScroll: true });
        readerTitle.closest("article").scrollIntoView();
      }
    });
    row.append(button);
    return row;
  });
  documentList.replaceChildren(...rows);
}

async function loadDocuments(preferredId) {
  state.pageRequest?.abort();
  const request = new AbortController();
  state.pageRequest = request;
  clearReader("Choose a document to read.");
  previous.disabled = true;
  next.disabled = true;
  pageNumber.textContent = "";
  retryList.hidden = true;
  listSkeleton.hidden = false;
  documentList.replaceChildren();
  setStatus(catalogStatus, "Loading documents…");

  try {
    const page = await listDocuments({
      nikaya: filter.value,
      limit: PAGE_SIZE,
      offset: state.offset,
      signal: request.signal,
    });
    if (request.signal.aborted) return;
    if (state.offset > 0 && state.offset >= page.total) {
      state.offset = page.total ? Math.floor((page.total - 1) / PAGE_SIZE) * PAGE_SIZE : 0;
      if (page.total) return loadDocuments(preferredId);
    }
    listSkeleton.hidden = true;
    renderDocuments(page.items);
    if (!page.items.length) {
      setStatus(catalogStatus, "No documents match this filter.");
      updateUrl();
      return;
    }

    const first = page.offset + 1;
    const last = page.offset + page.items.length;
    setStatus(catalogStatus, first + "–" + last + " of " + page.total + " documents");
    pageNumber.textContent = "Page " + (page.offset / page.limit + 1);
    previous.disabled = page.offset === 0;
    next.disabled = last >= page.total;
    const selected = page.items.find((doc) => String(doc.id) === preferredId) ?? page.items[0];
    readDocument(selected);
  } catch {
    if (request.signal.aborted) return;
    listSkeleton.hidden = true;
    setStatus(catalogStatus, "Couldn’t load documents. Check that the API is running.", true);
    retryList.hidden = false;
  }
}

function readDocument(doc) {
  resetMetadata();
  state.selected = doc;
  updateUrl();
  for (const button of documentList.querySelectorAll("button")) {
    if (button.dataset.documentId === String(doc.id)) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }
  readerTitle.textContent = doc.title;
  readerCode.textContent = documentCode(doc) + " / " + doc.status;
  readerCode.hidden = false;
  details.hidden = false;
  loadChunks(doc);
  if (details.open) loadMetadata(doc);
}

function addField(list, label, value) {
  if (value == null || value === "") return;
  list.append(textElement("dt", "", label));
  const field = document.createElement("dd");
  field.append(value);
  list.append(field);
}

function renderMetadata(doc) {
  const fields = document.createElement("dl");
  fields.className = "metadata-fields";
  addField(fields, "Document ID", String(doc.id));
  addField(fields, "Status", doc.status);
  addField(fields, "Nikāya", doc.nikaya?.toUpperCase());
  addField(fields, "Translator", doc.author_uid);
  addField(fields, "Language", doc.lang);
  addField(fields, "Type", doc.doc_type?.replaceAll("_", " "));
  addField(fields, "Category", doc.category);
  if (doc.created_at) {
    addField(fields, "Added", dateFormatter.format(new Date(doc.created_at)));
  }
  addField(fields, "Tags", doc.tags.join(", "));
  if (doc.reading_url) {
    const source = textElement("a", "source-link", doc.source === "suttacentral" ? "SuttaCentral" : "Read original");
    source.href = doc.reading_url;
    addField(fields, "Source", source);
  } else {
    addField(fields, "Source", doc.source);
  }
  if (doc.description) detailsContent.append(textElement("p", "metadata-description", doc.description));
  detailsContent.append(fields);
}

async function loadMetadata(doc) {
  if (state.detailsLoaded || state.detailRequest) return;
  const request = new AbortController();
  state.detailRequest = request;
  retryDetails.hidden = true;
  detailsStatus.hidden = false;
  setStatus(detailsStatus, "Loading metadata…");

  try {
    const result = await getDocument(doc.id, request.signal);
    if (request.signal.aborted) return;
    renderMetadata(result);
    state.detailsLoaded = true;
    detailsStatus.hidden = true;
  } catch (error) {
    if (request.signal.aborted) return;
    const missing = error instanceof ApiError && error.status === 404;
    setStatus(detailsStatus, missing ? "This document is no longer available." : "Couldn’t load metadata.", true);
    retryDetails.hidden = missing;
  } finally {
    if (state.detailRequest === request) state.detailRequest = null;
  }
}

function renderChunk(item, position) {
  const section = document.createElement("section");
  section.className = "chunk";
  section.id = "chunk-" + (position + 1);
  const label = "Chunk " + (position + 1);
  const paths = item.headers.filter(Boolean);
  const heading = textElement("h3", "", label);
  heading.tabIndex = -1;
  section.append(heading);
  if (paths.length) section.append(textElement("p", "chunk-path", paths.join(" / ")));
  section.append(textElement("p", "chunk-text", item.text));

  const inspect = document.createElement("details");
  inspect.className = "chunk-metadata";
  const fields = document.createElement("dl");
  fields.className = "metadata-fields";
  addField(fields, "UUID", textElement("code", "chunk-uuid", item.uuid));
  addField(fields, "Index", String(item.chunk_index));
  inspect.append(textElement("summary", "", "Chunk metadata"), fields);
  section.append(inspect);
  const name = paths[0]?.split(" > ").at(-1);
  chunkSelect.add(new Option(name ? label + ": " + name : label, section.id));
  return section;
}

async function loadChunks(doc) {
  resetChunks();
  const request = new AbortController();
  state.chunkRequest = request;
  setStatus(readerStatus, "Loading chunks…");
  readerSkeleton.hidden = false;
  refreshChunks.hidden = false;
  refreshChunks.disabled = true;
  refreshChunks.textContent = "Refresh chunks";

  try {
    const result = await getChunks(doc.id, request.signal);
    if (request.signal.aborted) return;
    readerSkeleton.hidden = true;
    if (!result.items.length) {
      setStatus(readerStatus, "This document has no chunks yet.");
      return;
    }
    const count = result.items.length;
    setStatus(readerStatus, count + (count === 1 ? " chunk" : " chunks"));
    chunks.append(...result.items.map(renderChunk));
    chunkJump.hidden = count < 2;
  } catch (error) {
    if (request.signal.aborted) return;
    readerSkeleton.hidden = true;
    if (error instanceof ApiError && error.status === 404) {
      setStatus(readerStatus, "This document is no longer available. Choose another document.", true);
      refreshChunks.hidden = true;
    } else {
      setStatus(readerStatus, "Couldn’t load chunks. Try again.", true);
      refreshChunks.textContent = "Retry chunks";
    }
  } finally {
    if (!request.signal.aborted) refreshChunks.disabled = false;
  }
}

filter.addEventListener("change", () => {
  state.offset = 0;
  loadDocuments();
  updateUrl();
});
previous.addEventListener("click", () => {
  state.offset -= PAGE_SIZE;
  loadDocuments();
  updateUrl();
});
next.addEventListener("click", () => {
  state.offset += PAGE_SIZE;
  loadDocuments();
  updateUrl();
});
retryList.addEventListener("click", () => loadDocuments(new URLSearchParams(location.search).get("document")));
refreshChunks.addEventListener("click", () => {
  if (state.selected) loadChunks(state.selected);
});
details.addEventListener("toggle", () => {
  if (details.open && state.selected) loadMetadata(state.selected);
});
retryDetails.addEventListener("click", () => {
  if (state.selected) loadMetadata(state.selected);
});
chunkSelect.addEventListener("change", () => {
  const section = document.getElementById(chunkSelect.value);
  if (section) {
    section.querySelector("h3").focus({ preventScroll: true });
    section.scrollIntoView();
  }
});

loadDocuments(params.get("document"));
