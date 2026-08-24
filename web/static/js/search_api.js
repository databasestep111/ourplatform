/*
 * OurPlatform Search API Client
 *
 * Browser-side bridge:
 *
 * HTML
 *   ↓
 * JavaScript
 *   ↓
 * /api/search
 *   ↓
 * Python API
 *   ↓
 * search/search.py
 *   ↓
 * JSON
 *   ↓
 * JavaScript
 *   ↓
 * HTML
 */

"use strict";


// ============================================================================
// CONFIGURATION
// ============================================================================

const SearchAPI = (() => {

    const CONFIG = {
        baseURL: "/api/search",

        defaultLimit: 10,

        maxLimit: 500,

        requestTimeout: 30000,

        method: "POST",

        headers: {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    };


    // ========================================================================
    // INTERNAL STATE
    // ========================================================================

    const state = {
        loading: false,

        lastQuery: "",

        lastResponse: null,

        lastError: null,

        requestId: 0,

        activeController: null,

        initialized: false
    };


    // ========================================================================
    // UTILITIES
    // ========================================================================

    function normalizeQuery(query) {

        if (query === null || query === undefined) {
            return "";
        }

        return String(query).trim();
    }


    function normalizeCategory(category) {

        if (
            category === null ||
            category === undefined
        ) {
            return null;
        }

        const value = String(category).trim();

        return value || null;
    }


    function normalizeTags(tags) {

        if (!tags) {
            return [];
        }

        if (typeof tags === "string") {

            return tags
                .split(/[;,]/)
                .map(tag => tag.trim())
                .filter(Boolean);
        }

        if (Array.isArray(tags)) {

            return tags
                .map(tag => String(tag).trim())
                .filter(Boolean);
        }

        return [];
    }


    function normalizeLimit(limit) {

        let value = Number(limit);

        if (!Number.isFinite(value)) {
            value = CONFIG.defaultLimit;
        }

        value = Math.floor(value);

        if (value < 1) {
            value = 1;
        }

        if (value > CONFIG.maxLimit) {
            value = CONFIG.maxLimit;
        }

        return value;
    }


    function createRequestID() {

        state.requestId += 1;

        return state.requestId;
    }


    function createAbortController() {

        if (
            typeof AbortController === "undefined"
        ) {
            return null;
        }

        return new AbortController();
    }


    function createTimeout(controller) {

        if (!controller) {
            return null;
        }

        return setTimeout(() => {

            controller.abort();

        }, CONFIG.requestTimeout);
    }


    function clearTimeoutSafe(timeout) {

        if (timeout !== null) {
            clearTimeout(timeout);
        }
    }


    // ========================================================================
    // REQUEST BODY
    // ========================================================================

    function buildSearchPayload(options = {}) {

        return {
            query: normalizeQuery(
                options.query
            ),

            category: normalizeCategory(
                options.category
            ),

            tags: normalizeTags(
                options.tags
            ),

            limit: normalizeLimit(
                options.limit
            )
        };
    }


    // ========================================================================
    // RESPONSE NORMALIZATION
    // ========================================================================

    function normalizeResponse(payload) {

        if (!payload || typeof payload !== "object") {

            return {
                success: false,

                status: 500,

                message: "Invalid API response.",

                data: null,

                errors: []
            };
        }

        return {
            success: Boolean(
                payload.success
            ),

            status: Number(
                payload.status || 200
            ),

            message: (
                payload.message ||
                ""
            ),

            data: (
                payload.data ||
                null
            ),

            errors: Array.isArray(
                payload.errors
            )
                ? payload.errors
                : [],

            metadata: (
                payload.metadata &&
                typeof payload.metadata === "object"
            )
                ? payload.metadata
                : {}
        };
    }


    function extractResults(response) {

        if (
            !response ||
            !response.data
        ) {
            return [];
        }

        if (
            Array.isArray(
                response.data.results
            )
        ) {
            return response.data.results;
        }

        return [];
    }


    // ========================================================================
    // ERROR HANDLING
    // ========================================================================

    function createError(
        message,
        details = {}
    ) {

        const error = new Error(
            message
        );

        Object.assign(
            error,
            details
        );

        return error;
    }


    function apiErrorFromResponse(
        response
    ) {

        const message =
            response.message ||
            "Search request failed.";

        return createError(
            message,
            {
                apiResponse: response,

                status: response.status,

                errors: response.errors
            }
        );
    }


    // ========================================================================
    // CORE REQUEST
    // ========================================================================

    async function request(
        payload
    ) {

        const requestID =
            createRequestID();

        const controller =
            createAbortController();

        const timeout =
            createTimeout(
                controller
            );

        state.loading = true;

        state.lastError = null;

        state.activeController =
            controller;

        try {

            const response =
                await fetch(
                    CONFIG.baseURL,
                    {
                        method: CONFIG.method,

                        headers: {
                            ...CONFIG.headers
                        },

                        body: JSON.stringify(
                            payload
                        ),

                        signal:
                            controller
                            ? controller.signal
                            : undefined
                    }
                );


            let json;

            try {

                json =
                    await response.json();

            } catch (error) {

                throw createError(
                    "The API returned an invalid response.",
                    {
                        status:
                            response.status,

                        cause: error
                    }
                );
            }


            const normalized =
                normalizeResponse(
                    json
                );


            normalized.httpStatus =
                response.status;


            if (
                !response.ok ||
                !normalized.success
            ) {

                throw apiErrorFromResponse(
                    normalized
                );
            }


            state.lastResponse =
                normalized;


            return normalized;

        } catch (error) {

            state.lastError =
                error;

            throw error;

        } finally {

            clearTimeoutSafe(
                timeout
            );

            if (
                state.requestId ===
                requestID
            ) {

                state.loading = false;

                state.activeController =
                    null;
            }
        }
    }


    // ========================================================================
    // SEARCH
    // ========================================================================

    async function search(
        options = {}
    ) {

        const payload =
            buildSearchPayload(
                options
            );


        if (!payload.query) {

            throw createError(
                "Please enter a search query."
            );
        }


        state.lastQuery =
            payload.query;


        return request(
            payload
        );
    }


    // ========================================================================
    // SEARCH WITH CALLBACKS
    // ========================================================================

    async function searchSafe(
        options = {}
    ) {

        try {

            const response =
                await search(
                    options
                );

            return {
                success: true,

                response,

                results:
                    extractResults(
                        response
                    ),

                error: null
            };

        } catch (error) {

            return {
                success: false,

                response: null,

                results: [],

                error
            };
        }
    }


    // ========================================================================
    // TITLE SEARCH
    // ========================================================================

    async function searchTitle(
        query,
        limit = CONFIG.defaultLimit
    ) {

        return specializedSearch(
            "/title",
            {
                query:
                    normalizeQuery(
                        query
                    ),

                limit:
                    normalizeLimit(
                        limit
                    )
            }
        );
    }


    // ========================================================================
    // CONTENT SEARCH
    // ========================================================================

    async function searchContent(
        query,
        limit = CONFIG.defaultLimit
    ) {

        return specializedSearch(
            "/content",
            {
                query:
                    normalizeQuery(
                        query
                    ),

                limit:
                    normalizeLimit(
                        limit
                    )
            }
        );
    }


    // ========================================================================
    // CATEGORY SEARCH
    // ========================================================================

    async function searchCategory(
        category
    ) {

        const value =
            normalizeCategory(
                category
            );

        if (!value) {

            throw createError(
                "A category is required."
            );
        }


        return specializedSearch(
            "/category",
            {
                category: value
            }
        );
    }


    // ========================================================================
    // TAG SEARCH
    // ========================================================================

    async function searchTag(
        tag
    ) {

        const tags =
            normalizeTags(
                tag
            );

        if (!tags.length) {

            throw createError(
                "A tag is required."
            );
        }


        return specializedSearch(
            "/tag",
            {
                tag: tags[0]
            }
        );
    }


    // ========================================================================
    // SPECIALIZED REQUESTS
    // ========================================================================

    async function specializedSearch(
        endpoint,
        body
    ) {

        const originalURL =
            CONFIG.baseURL;


        const url =
            originalURL +
            endpoint;


        const controller =
            createAbortController();

        const timeout =
            createTimeout(
                controller
            );


        try {

            const response =
                await fetch(
                    url,
                    {
                        method: "POST",

                        headers: {
                            ...CONFIG.headers
                        },

                        body: JSON.stringify(
                            body
                        ),

                        signal:
                            controller
                            ? controller.signal
                            : undefined
                    }
                );


            const json =
                await response.json();


            const normalized =
                normalizeResponse(
                    json
                );


            if (
                !response.ok ||
                !normalized.success
            ) {

                throw apiErrorFromResponse(
                    normalized
                );
            }


            return normalized;

        } finally {

            clearTimeoutSafe(
                timeout
            );
        }
    }


    // ========================================================================
    // ITEM
    // ========================================================================

    async function getItem(
        id
    ) {

        const numericID =
            Number(id);

        if (
            !Number.isInteger(
                numericID
            ) ||
            numericID < 1
        ) {

            throw createError(
                "A valid item ID is required."
            );
        }


        return specializedSearch(
            "/item",
            {
                id: numericID
            }
        );
    }


    // ========================================================================
    // STATISTICS
    // ========================================================================

    async function getStatistics() {

        return specializedSearch(
            "/statistics",
            {}
        );
    }


    // ========================================================================
    // COUNT
    // ========================================================================

    async function getCount() {

        return specializedSearch(
            "/count",
            {}
        );
    }


    // ========================================================================
    // CATEGORIES
    // ========================================================================

    async function getCategories() {

        return specializedSearch(
            "/categories",
            {}
        );
    }


    // ========================================================================
    // TAGS
    // ========================================================================

    async function getTags() {

        return specializedSearch(
            "/tags",
            {}
        );
    }


    // ========================================================================
    // DUPLICATE CHECK
    // ========================================================================

    async function checkDuplicate(
        content
    ) {

        const value =
            normalizeQuery(
                content
            );

        if (!value) {

            throw createError(
                "Content is required."
            );
        }


        return specializedSearch(
            "/duplicate",
            {
                content: value
            }
        );
    }


    // ========================================================================
    // CANCEL
    // ========================================================================

    function cancel() {

        if (
            state.activeController
        ) {

            state.activeController.abort();

            state.activeController =
                null;

            state.loading = false;
        }
    }


    // ========================================================================
    // STATE
    // ========================================================================

    function isLoading() {

        return state.loading;
    }


    function getLastQuery() {

        return state.lastQuery;
    }


    function getLastResponse() {

        return state.lastResponse;
    }


    function getLastError() {

        return state.lastError;
    }


    function getState() {

        return {
            loading:
                state.loading,

            lastQuery:
                state.lastQuery,

            lastResponse:
                state.lastResponse,

            lastError:
                state.lastError,

            initialized:
                state.initialized
        };
    }


    // ========================================================================
    // CONFIGURATION ACCESS
    // ========================================================================

    function getConfig() {

        return {
            ...CONFIG,

            headers: {
                ...CONFIG.headers
            }
        };
    }


    // ========================================================================
    // INITIALIZATION
    // ========================================================================

    function initialize(
        options = {}
    ) {

        if (
            state.initialized
        ) {
            return getState();
        }


        if (
            options.baseURL
        ) {

            CONFIG.baseURL =
                String(
                    options.baseURL
                ).replace(
                    /\/$/,
                    ""
                );
        }


        if (
            options.timeout
        ) {

            const timeout =
                Number(
                    options.timeout
                );

            if (
                Number.isFinite(
                    timeout
                ) &&
                timeout > 0
            ) {

                CONFIG.requestTimeout =
                    timeout;
            }
        }


        if (
            options.defaultLimit
        ) {

            CONFIG.defaultLimit =
                normalizeLimit(
                    options.defaultLimit
                );
        }


        state.initialized =
            true;


        return getState();
    }


    // ========================================================================
    // PUBLIC API
    // ========================================================================

    return {

        initialize,

        search,

        searchSafe,

        searchTitle,

        searchContent,

        searchCategory,

        searchTag,

        getItem,

        getStatistics,

        getCount,

        getCategories,

        getTags,

        checkDuplicate,

        cancel,

        isLoading,

        getLastQuery,

        getLastResponse,

        getLastError,

        getState,

        getConfig
    };

})();


// ============================================================================
// GLOBAL AVAILABILITY
// ============================================================================

if (
    typeof window !== "undefined"
) {

    window.SearchAPI =
        SearchAPI;

}


// ============================================================================
// AUTOMATIC INITIALIZATION
// ============================================================================

if (
    typeof document !== "undefined"
) {

    if (
        document.readyState ===
        "loading"
    ) {

        document.addEventListener(
            "DOMContentLoaded",
            () => {

                SearchAPI.initialize();

            },
            {
                once: true
            }
        );

    } else {

        SearchAPI.initialize();
    }
}