# pure.md api (1.0.0)

Download OpenAPI specification:[Download](blob:https://pure.md/543dbb49-5629-4c03-8d0b-efa34e22ae5d)

Support: [puremd@crawlspace.dev](mailto:puremd@crawlspace.dev) [Terms of Service](https://pure.md/terms)

## [](https://pure.md/docs/#section/Introduction)Introduction

[pure.md](https://pure.md/) is a REST API that lets AI agents and developers reliably access web content. With pure.md, you can:

-   Avoid bot detection by mimicking real user behavior
-   Render JavaScript-heavy websites, PDFs, images, and files
-   Scrape web pages into markdown optimized for an LLM
-   Crawl search engines for up-to-date knowledge
-   Extract JSON from web pages using natural language

## [](https://pure.md/docs/#section/Authentication)Authentication

Generate a unique API token in your [dashboard](https://pure.md/dashboard). Then include that token in the `x-puremd-api-token` request header for all requests.

## [](https://pure.md/docs/#section/Rate-limits)Rate limits

Rate limits vary by subscription plan. See [pricing](https://pure.md/#pricing) for details.

| Subscription type | Requests per minute |
| --- | --- |
| Logged out / anonymous | 6   |
| Logged in, no subscription | 10  |
| Starter plan | 60  |
| Growth plan | 600 |
| Business plan | 3000 |

## [](https://pure.md/docs/#section/MCP-server)MCP server

The [Model Context Protocol](https://modelcontextprotocol.io/introduction), developed by Anthropic, is an open standard that enables AI systems to seamlessly interact with an ecosystem of tooling. With it, MCP clients like Cursor, Windsurf, and Claude Desktop can learn how to use a variety of APIs and other functionality.

You can instruct your MCP clients to route traffic through pure.md by following the instructions at [https://github.com/puremd/puremd-mcp](https://github.com/puremd/puremd-mcp).

## [](https://pure.md/docs/#section/Headers-pass-through)Headers pass through

All request headers pass through to the target URL, except ones that begin with `x-puremd-`.

Original headers from the origin are returned in the response.

* * *

## [](https://pure.md/docs/#/paths/~1:url/get)Fetch web content

Retrieves the content of a given URL in markdown format. Use this endpoint to scrape text content from a web page without getting blocked.

##### Authorizations:

_APIToken_

##### path Parameters

| url required | string The URL |
| --- | --- |

### Responses

**200**

OK

**400**

Bad request

**415**

Unsupported media type

**429**

Rate limit exceeded

get/:url

Production

https://pure.md/:url

### Response samples

-   200
-   400
-   415
-   429

Content type

text/plain

Copy

<WebPage url\="https://example.com"\>

title: Example Domain
access\_date: Wed, 05 Mar 2025 22:27:19 GMT

\--\-

# Example Domain

This domain is for use in illustrative examples in documents. You may use this domain in literature without prior coordination or asking for permission.

More information...

</WebPage\>

## [](https://pure.md/docs/#/paths/~1:url/post)Fetch and extract data

**This endpoint is only available on paid plans.**

Runs inference on the content of a given URL. Use this endpoint to extract structured JSON from a webpage.

##### Authorizations:

_APIToken_

##### path Parameters

| url required | string The URL |
| --- | --- |

##### Request Body schema: application/json

| prompt required | string The user message |
| --- | --- |
| model | string Enum: "meta/llama-3.1-8b" … 3 more The generative AI model to use. Smaller models are faster, while larger models are more accurate. Default model: `meta/llama-3.1-8b` |
| schema | object JSON schema of the desired response. Omit this property to get a response in plaintext. |

### Responses

**200**

OK

**400**

Bad request

**401**

Unauthorized

**402**

Payment required

**415**

Unsupported media type

**429**

Rate limit exceeded

post/:url

Production

https://pure.md/:url

### Request samples

-   Payload

Content type

application/json

Copy

Expand all Collapse all

`{  -   "prompt": "What are the top 5 headlines from today?",      -   "model": "meta/llama-3.1-8b",      -   "schema": {          -   "type": "object",              -   "properties": {                  -   "headlines": {                          -   "type": "array",                              -   "items": {                                  -   "type": "string"                                                       }                                           }                               },              -   "required": [                  -   "headlines"                               ]                   }       }`

### Response samples

-   200
-   400
-   401
-   402
-   415
-   429

Content type

application/jsontext/plainapplication/json

Copy

Expand all Collapse all

`{  -   "type": "object",      -   "properties": {          -   "headlines": {                  -   "type": "array",                      -   "items": {                          -   "type": "string"                                           }                               }                   },      -   "required": [          -   "headlines"                   ]       }`

## [](https://pure.md/docs/#/paths/~1search?q=/get)Search the web

**This endpoint is only available on paid plans.**

Crawls the top results from a search engine query and concatenates the web content from all pages into markdown. Use this endpoint to gather knowledge of news, current events, or specific topics.

##### Authorizations:

_APIToken_

##### query Parameters

| q required | string The URL-encoded search query |
| --- | --- |

### Responses

**200**

OK

**400**

Bad request

**401**

Unauthorized

**402**

Payment required

**429**

Rate limit exceeded

get/search?q=

Production

https://pure.md/search?q=

### Response samples

-   200
-   400
-   401
-   402
-   429

Content type

text/plain

Copy

\# Title of the Page

## Introduction
This is the introduction text from the webpage, purified and optimized for LLM processing.

## Main Content
The main content of the page converted to clean markdown format, with unnecessary elements removed.

### Subsection
Content organized in logical subsections with proper hierarchy.

## Conclusion
The concluding information from the webpage.

## [](https://pure.md/docs/#/paths/~1search?q=/post)Search and extract data

**This endpoint is only available on paid plans.**

Crawls the top results from a search engine query and runs inference on their web content. Use this endpoint to answer questions about news, current events, or general user queries that require searching.

##### Authorizations:

_APIToken_

##### path Parameters

| url required | string The URL |
| --- | --- |

##### Request Body schema: application/json

| prompt required | string The user message |
| --- | --- |
| model | string Enum: "meta/llama-3.1-8b" … 3 more The generative AI model to use. Smaller models are faster, while larger models are more accurate. Default model: `meta/llama-3.1-8b` |
| schema | object JSON schema of the desired response. Omit this property to get a response in plaintext. |

### Responses

**200**

OK

**400**

Bad request

**401**

Unauthorized

**402**

Payment required

**429**

Rate limit exceeded

post/search?q=

Production

https://pure.md/search?q=

### Request samples

-   Payload

Content type

application/json

Copy

Expand all Collapse all

`{  -   "prompt": "Who won the baseball game last night?",      -   "model": "meta/llama-3.1-8b",      -   "schema": {          -   "type": "object",              -   "properties": {                  -   "headlines": {                          -   "type": "array",                              -   "items": {                                  -   "type": "string"                                                       }                                           }                               },              -   "required": [                  -   "headlines"                               ]                   }       }`

### Response samples

-   200
-   400
-   401
-   402
-   429

Content type

application/jsontext/plainapplication/json

Copy

`"string"`
