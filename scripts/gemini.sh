#!/usr/bin/env bash
set -euo pipefail

# Configuration
PROJECT_ID="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null || echo "")}"
MODEL_ID="${GEMINI_MODEL:-gemini-3.8-flash}"
LOCATION="${GEMINI_LOCATION:-global}"

if [ -z "${PROJECT_ID}" ] || [ "${PROJECT_ID}" = "(unset)" ]; then
  echo "Error: No active GCP Project configured. Set GCP_PROJECT or run 'gcloud config set project <PROJECT_ID>'." >&2
  exit 1
fi

# Read prompt from arguments or stdin
if [ "$#" -gt 0 ]; then
  PROMPT="$*"
elif [ ! -t 0 ]; then
  PROMPT="$(cat -)"
else
  echo "Usage: $0 \"<your prompt>\"" >&2
  echo "   or: cat file.txt | $0 \"<instructions>\"" >&2
  exit 1
fi

TOKEN="$(gcloud auth print-access-token)"

if [ -z "${TOKEN}" ]; then
  echo "Error: Failed to obtain gcloud access token. Please run 'gcloud auth login'." >&2
  exit 1
fi

ENDPOINT="https://aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:generateContent"

PAYLOAD="$(jq -n --arg text "${PROMPT}" '{contents: [{role: "user", parts: [{text: $text}]}]}')"

RESPONSE="$(curl -s -X POST \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  "${ENDPOINT}" \
  -d "${PAYLOAD}")"

if echo "${RESPONSE}" | jq -e '.error' >/dev/null 2>&1; then
  echo "Error from Vertex AI:" >&2
  echo "${RESPONSE}" | jq '.error.message' >&2
  exit 1
fi

echo "${RESPONSE}" | jq -r '.candidates[0].content.parts[0].text // "No text returned."'
