#!/usr/bin/env bash
# Prints a "Contributors" section for release notes, thanking everyone who authored a merged
# pull request, or an issue closed by one, between two refs. The maintainer and bots are left
# out. Prints nothing if nobody is left.
#
# Usage: release_contributors.sh <previous-ref> <target-ref>
# Needs GITHUB_REPOSITORY (owner/name) and a GH_TOKEN that gh can use.
set -euo pipefail

REPO="${GITHUB_REPOSITORY:?set GITHUB_REPOSITORY to owner/name}"
FROM="$1"
TO="$2"
OWNER="${REPO%%/*}"
NAME="${REPO#*/}"

prs=$(for sha in $(git rev-list "$FROM..$TO"); do
  gh api "repos/$REPO/commits/$sha/pulls" --jq '.[] | select(.merged_at != null) | .number'
done | sort -un)

logins=$(for pr in $prs; do
  gh api graphql -F owner="$OWNER" -F name="$NAME" -F number="$pr" -f query='
    query($owner: String!, $name: String!, $number: Int!) {
      repository(owner: $owner, name: $name) {
        pullRequest(number: $number) {
          author { login }
          closingIssuesReferences(first: 20) { nodes { author { login } } }
        }
      }
    }' --jq '.data.repository.pullRequest
             | [.author.login] + [.closingIssuesReferences.nodes[].author.login]
             | .[] | select(. != null)'
done | sort -fu)

thanked=()
for login in $logins; do
  case "$login" in
    "$OWNER" | *\[bot\] | Copilot | copilot-* | github-actions) continue ;;
  esac
  thanked+=("@$login")
done

if [ "${#thanked[@]}" -eq 0 ]; then
  exit 0
fi

if [ "${#thanked[@]}" -eq 1 ]; then
  names="${thanked[0]}"
else
  last="${thanked[${#thanked[@]}-1]}"
  rest=("${thanked[@]:0:${#thanked[@]}-1}")
  names="$(printf '%s, ' "${rest[@]}")"
  names="${names%, } and $last"
fi

printf '\n### Contributors\n\nThanks to %s for the issues and pull requests that went into this release.\n' "$names"
