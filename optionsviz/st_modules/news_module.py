"""
News Analysis Module

This module handles all news-related functionality for the Financial Analysis Dashboard.
It includes functions for news retrieval, processing, and AI analysis.

Author: Mithul Raaj
Created: March 2025
License: MIT
"""

# Standard library imports
import math
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple

# Third-party imports
import streamlit as st
import pinecone

# Try to import news-specific modules
# try:
from news.news_core import fetch_news, process_news_articles_whole
from news.embedding_utils import (
    create_pinecone_index,
    clear_pinecone_index,
    upload_news_to_pinecone,
    preprocess_query,
    NewsRetriever
)
from news.llm_interface import LLMInterface as NewsLLM

# Initialize module availability flag
# NEWS_MODULES_AVAILABLE = False
# Set flag if all imports are successful
NEWS_MODULES_AVAILABLE = True
# except ImportError:
    # Keep flag as False if import fails
    # pass


def change_page(direction: str) -> None:
    """
    Function to change the current page for news articles pagination.

    Args:
        direction (str): Direction to change the page ('next' or 'prev')
    """
    if direction == "next":
        st.session_state['current_page'] += 1
    else:
        st.session_state['current_page'] -= 1

@st.cache_resource
def get_retriever(pinecone_api_key: str,
                  index_name: str, model_name: str) -> Optional[NewsRetriever]:
    """
    Initialize and cache the news retriever to avoid recreating it on every rerun.

    Args:
        pinecone_api_key (str): API key for Pinecone
        index_name (str): Name of the Pinecone index to use
        model_name (str): Name of the embedding model to use

    Returns:
        Optional[NewsRetriever]: Initialized retriever or None if initialization failed
    """
    if not pinecone_api_key:
        return None

    try:
        # Create a dictionary mapping models to their dimensions
        model_dimensions = {
            "intfloat/e5-large-v2": 1024,
            "intfloat/e5-base-v2": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "BAAI/bge-base-v1.5": 768
        }

        # Check if the Pinecone index exists and get its dimension
        pc = pinecone.Pinecone(api_key=pinecone_api_key)

        # Try to get index info to check its dimension
        index_info = pc.describe_index(name=index_name)
        index_dimension = index_info.dimension

        # Select a model with matching dimension
        if model_name not in model_dimensions or model_dimensions[model_name] !=index_dimension:
            # Find a model with matching dimension
            matching_models = [m for m,
                                dim in model_dimensions.items() if dim == index_dimension]
            if matching_models:
                # Use the first matching model
                st.warning(f"Selected dimension({model_dimensions.get(model_name, 'unknown')}) "
                            f"doesn't match index dimension ({index_dimension}). "
                            f"Using {matching_models[0]} instead.")
                model_name = matching_models[0]
            else:
                st.error(f"No compatible model found for index dimension {index_dimension}. "
                        f"You need to recreate your index with dimension"
                        f"{model_dimensions.get(model_name, 2048)}.")
                return None

        retriever = NewsRetriever(
            pinecone_api_key=pinecone_api_key,
            index_name=index_name,
            model_name=model_name
        )
        return retriever
    except (ValueError, KeyError) as e:
        st.error(f"Error initializing retriever: {str(e)}")
        return None
    except ImportError:
        st.error("Pinecone library not available. Install with: pip install pinecone-client")
        return None


@st.cache_resource
def get_llm(api_key: str, model_name: str) -> Optional[NewsLLM]:
    """
    Initialize and cache the LLM interface to avoid recreating it on every rerun.

    Args:
        api_key (str): API key for the LLM service
        model_name (str): Name of the LLM model to use

    Returns:
        Optional[NewsLLM]: Initialized LLM interface or None if initialization failed
    """
    if not api_key:
        return None

    try:
        llm = NewsLLM(
            api_key=api_key,
            model=model_name
        )
        return llm
    except (ValueError, KeyError):
        return None


def render_news_sidebar(
    pinecone_api_key: str,
    openrouter_api_key: str,
    newsapi_key: str
) -> Tuple[Optional[NewsRetriever], Optional[NewsLLM]]:
    """
    Render the sidebar for the News Analyzer application.

    Args:
        pinecone_api_key (str): API key for Pinecone
        openrouter_api_key (str): API key for OpenRouter
        newsapi_key (str): API key for News API

    Returns:
        tuple: (retriever, llm) - The initialized news retriever and LLM interface
    """
    st.sidebar.header("News Analyzer Settings")

    # Pinecone settings with unique keys
    index_name = st.sidebar.text_input("Pinecone Index Name", "newsdata", key="news_index_name")

    # Model selection with dimension information
    model_info = [
        {"name": "intfloat/e5-large-v2", "dim": 1024, "info": "High quality, 1024 dimensions"},
        {"name": "intfloat/e5-base-v2", "dim": 768, "info": "Good balance, 768 dimensions"},
        {"name": "BAAI/bge-large-en-v1.5", "dim": 1024, "info": "High quality, 1024 dimensions"},
        {"name": "BAAI/bge-base-v1.5", "dim": 768, "info": "Good balance, 768 dimensions"}
    ]

    model_options = [f"{m['name']} ({m['info']})" for m in model_info]
    selected_model_idx = st.sidebar.selectbox(
        "Embedding Model",
        options=range(len(model_options)),
        format_func=lambda i: model_options[i],
        index=0,
        key="news_embedding_model_idx"
    )

    # Extract the actual model name from the selection
    embedding_model = model_info[selected_model_idx]["name"]
    st.session_state["news_embedding_model"] = embedding_model

    # Show dimension information
    st.sidebar.caption(f"Model dimension: {model_info[selected_model_idx]['dim']}")

    # LLM settings
    llm_model = st.sidebar.selectbox(
        "LLM Model",
        ["deepseek/deepseek-chat:free", "google/gemini-pro",
         "anthropic/claude-3-sonnet", "meta-llama/llama-3-8b-instruct"],
        index=0,
        key="news_llm_model"
    )

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        st.checkbox("Show Query Processing", value=False, key="news_show_query")
        st.slider("Number of Results", min_value=3, max_value=20, value=10,
                  key="news_results_count")

        # Option to create/recreate index with matching dimensions
        if st.button("Create New Index With Matching Dimensions"):
            create_matching_index(pinecone_api_key,
                                  index_name, model_info[selected_model_idx]["dim"])

    # Initialize News services
    retriever = get_retriever(pinecone_api_key, index_name, embedding_model)
    llm = get_llm(openrouter_api_key, llm_model)

    # API key warnings
    if not pinecone_api_key:
        st.sidebar.error("PINECONE_API_KEY not set. Required for vector database.")
    if not openrouter_api_key:
        st.sidebar.error("OPENROUTER_API_KEY not set. Required for AI analysis.")
    if not newsapi_key:
        st.sidebar.error("NEWS_API_KEY not set. Required for fetching news.")

    return retriever, llm


def create_matching_index(api_key, index_name, dimension):
    """Create a new Pinecone index with dimensions matching the selected model."""
    if not api_key:
        st.error("Pinecone API key is required to create an index")
        return

    pc = pinecone.Pinecone(api_key=api_key)

    # Check if index exists
    try:
        # Try to describe the index
        existing_index = pc.describe_index(name=index_name)

        # If we got here, index exists - check if dimensions match
        if existing_index.dimension == dimension:
            st.success(f"'{index_name}' exists with matching dimension ({dimension})")
            return

        # If dimensions don't match, ask for confirmation to delete
        st.warning(f"Index '{index_name}' exists but has dimension {existing_index.dimension} "
                    f"instead of required {dimension}")

        # Ask for confirmation
        if st.button(f"Delete and recreate index '{index_name}' with dimension {dimension}"):
            pc.delete_index(name=index_name)
            st.info(f"Deleted index '{index_name}'")

            # Wait a moment for the deletion to complete
            time.sleep(3)

            # Create new index
            # pylint: disable=no-value-for-parameter
            # This is vectordb specific and the function signature is fixed
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            st.success(f"Created new index '{index_name}' with dimension {dimension}")

    except pinecone.exceptions.NotFoundException as e:
        # Index doesn't exist, create it
        if "not found" in str(e).lower():
            # pylint: disable=no-value-for-parameter
            # This is vectordb specific and the function signature is fixed
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
            st.success(f"Created new index '{index_name}' with dimension {dimension}")
        else:
            raise e

def render_news_app(retriever=None, llm=None) -> None:
    """
    Render the News Analyzer application.
    This includes tabs for fetching news and asking AI-powered questions about the news.

    Args:
        retriever: The initialized news retriever
        llm: The initialized LLM interface
    """
    # st.image("./images/header_img.png", use_container_width=True)

    # Set up the tabs
    tab1, tab2 = st.tabs(["NewsViz", "Ask Questions"])

    # Tab 1: Fetch News
    with tab1:
        render_news_fetch_tab()

    # Tab 2: Ask Questions
    with tab2:
        render_news_questions_tab(retriever, llm)


def render_news_fetch_tab() -> None:
    """
    Render the News Fetch tab content.
    This tab allows users to search for news articles, view them,
    and process them into a vector database.
    """
    st.markdown("#### Fetch News Articles")

    try:
        _render_news_search_form()
        _display_fetched_articles()
        _render_vector_db_section()
    except ImportError as e:
        st.error(f"Error: Could not import required modules: {str(e)}")
        st.error("Please check that all are installed and properly configured.")


def _render_news_search_form() -> None:
    """Render the news search form with all input parameters."""
    with st.form(key="news_fetch_form"):
        # News query parameters
        query = st.text_input("Search Query (required)",
                              "artificial intelligence",
                              key="news_search_query")

        col1, col2 = st.columns(2)
        with col1:
            _render_category_language_selectors()

        with col2:
            _render_date_selectors()

        sources = st.text_input("News Sources (comma-separated)", "",
                                key="news_sources_input")

        # Submit button
        fetch_submit = st.form_submit_button("Fetch News")

    # Process form submission (outside the form)
    if fetch_submit:
        _process_news_search_form(query, sources)


def _render_category_language_selectors() -> None:
    """Render the category and language selection widgets."""
    st.selectbox(
        "Category (for top headlines)",
        [None, "business", "entertainment",
         "general", "health", "science",
         "sports", "technology"],
        index=0,
        key="news_category_select"
    )
    st.selectbox(
        "Language",
        ["en", "es", "fr", "de", "it"],
        index=0,
        key="news_language_select"
    )


def _render_date_selectors() -> None:
    """Render the date selection widgets."""
    st.date_input(
        "From Date",
        datetime.now() - timedelta(days=7),
        key="news_from_date"
    )
    st.date_input(
        "To Date",
        datetime.now(),
        key="news_to_date"
    )


def _process_news_search_form(query, sources) -> None:
    """Process the news search form submission."""
    if not query:
        st.error("Search query is required.")
        return

    # Get form values from session state
    language = st.session_state.get("news_language_select", "en")
    category = st.session_state.get("news_category_select", None)
    from_date = st.session_state.get("news_from_date",
                                     datetime.now() - timedelta(days=7))
    to_date = st.session_state.get("news_to_date", datetime.now())

    # Fetch news articles
    with st.spinner(f"Fetching news for query: '{query}'..."):
        articles = fetch_news(
            query=query,
            sources=sources if sources else None,
            from_date=from_date.strftime('%Y-%m-%d'),
            to_date=to_date.strftime('%Y-%m-%d'),
            language=language,
            category=category
        )

        # Store in session state
        st.session_state['fetched_articles'] = articles
        st.session_state['current_page'] = 1

        if not articles:
            st.warning("No articles found.")
        else:
            st.success(f"Fetched {len(articles)} articles.")


def _display_fetched_articles() -> None:
    """Display the fetched articles with pagination."""
    if not st.session_state.get('fetched_articles'):
        return

    articles = st.session_state['fetched_articles']
    total_articles = len(articles)
    articles_per_page = 10
    total_pages = max(1, math.ceil(total_articles / articles_per_page))

    _render_pagination_controls(total_pages)
    _render_current_page_articles(articles, articles_per_page, total_articles)


def _render_pagination_controls(total_pages) -> None:
    """Render the pagination controls."""
    prev_col, page_col, next_col = st.columns([1, 3, 1])

    # Previous button on left
    with prev_col:
        prev_disabled = st.session_state['current_page'] <= 1
        st.button("← Previous", disabled=prev_disabled, key="prev_btn",
                on_click=change_page, args=("prev",), use_container_width=True)

    # Page counter in center
    with page_col:
        st.markdown(
            f"<div style='text-align: center'>Page {st.session_state['current_page']}"
            f" of {total_pages}</div>",
            unsafe_allow_html=True
        )

    # Next button on right
    with next_col:
        next_disabled = st.session_state['current_page'] >= total_pages
        st.button("Next →", disabled=next_disabled, key="next_btn",
                on_click=change_page, args=("next",), use_container_width=True)


def _render_current_page_articles(articles, articles_per_page, total_articles) -> None:
    """Render the articles for the current page."""
    current_page = st.session_state['current_page']
    start_idx = (current_page - 1) * articles_per_page
    end_idx = min(start_idx + articles_per_page, total_articles)

    # Display articles
    for i in range(start_idx, end_idx):
        _render_article(articles[i])

    st.caption(f"Showing {start_idx + 1}-{end_idx} of {total_articles} articles")


def _render_article(article) -> None:
    """Render a single article."""
    col1, col2 = st.columns([1, 3])

    with col1:
        _render_article_image(article)

    with col2:
        _render_article_details(article)

    st.divider()


def _render_article_image(article) -> None:
    """Render the article image or placeholder."""
    # Access placeholder constants from session state
    placeholder_image = st.session_state.get('PLACEHOLDER_IMAGE')
    placeholder_loaded = st.session_state.get('PLACEHOLDER_LOADED', False)

    if article.get('urlToImage') and article['urlToImage'].strip():
        try:
            st.image(article['urlToImage'], use_container_width=True)
        except (KeyError, ValueError):
            # If article image fails, use our preloaded placeholder
            if placeholder_loaded:
                st.image(placeholder_image, use_container_width=True)
    else:
        # No article image at all
        if placeholder_loaded:
            st.image(placeholder_image, use_container_width=True)


def _render_article_details(article) -> None:
    """Render the article details (title, source, etc.)."""
    st.markdown(f"### {article.get('title', 'No Title')}")

    source_name = "Unknown Source"
    if 'source' in article and isinstance(article['source'], dict):
        source_name = article['source'].get('name', 'Unknown Source')

    published = article.get('publishedAt', 'Unknown Date')
    st.markdown(f"**Source:** {source_name} | **Published:** {published}")
    st.markdown(article.get('description', 'No description available'))

    if article.get('url'):
        st.markdown(f"[Read more]({article['url']})")


def _render_vector_db_section() -> None:
    """Render the vector database section for processing articles."""
    if not st.session_state.get('fetched_articles'):
        return

    st.subheader("Store In Vector Database")
    st.write("Process articles and store them in the vector database")

    index_name = st.session_state.get("news_index_name", "newsdata")
    embedding_model = st.session_state.get("news_embedding_model",
                                           "intfloat/e5-large-v2")

    if st.button("Process and Store in Vector Database", key="process_store_btn"):
        _process_articles_for_vector_db(index_name, embedding_model)


def _process_articles_for_vector_db(index_name, embedding_model) -> None:
    """Process articles and store them in the vector database."""
    # Get API key from session state
    pinecone_api_key = st.session_state.get('PINECONE_API_KEY', '')

    articles = st.session_state.get('fetched_articles', [])

    # Process articles
    with st.spinner("Processing articles..."):
        chunks = process_news_articles_whole(articles)
        st.success(f"Created {len(chunks)} documents (one per article).")

    # Set up Vector DB
    with st.spinner("Setting up vector database..."):
        message = create_pinecone_index(
            pinecone_api_key=pinecone_api_key,
            index_name=index_name
        )
        st.info(message)

    # Clear existing data
    with st.spinner("Clearing previous data from vector database..."):
        message = clear_pinecone_index(pinecone_api_key, index_name)
        st.info(message)

    # Upload to Vector DB
    with st.spinner("Uploading to vector database..."):
        message = upload_news_to_pinecone(
                document_chunks=chunks,
                pinecone_api_key=pinecone_api_key,
                index_name=index_name,
                model_name=embedding_model
            )

        if "Error:" in message:
            st.error(message)
        else:
            st.success(message)
            st.success("✅ News articles have been processed!")
            st.info("Ask Questions' tab to analyze these articles!")


def render_news_questions_tab(retriever: Optional[NewsRetriever],
                          llm: Optional[NewsLLM]) -> None:
    """
    Render the News Questions tab content.
    This tab allows users to ask questions about the fetched news articles using AI.

    Args:
        retriever: The initialized news retriever
        llm: The initialized LLM interface
    """
    st.header("Ask Questions About the News")

    # Display chat history and handle user input
    _render_chat_history()
    _handle_user_question(retriever, llm)


def _render_chat_history() -> None:
    """Render the chat history."""
    for question, answer, sources in st.session_state.get('chat_history', []):
        st.chat_message("user").write(question)

        with st.chat_message("assistant"):
            st.write(answer)
            _render_sources_expander(sources)


def _render_sources_expander(sources) -> None:
    """Render the sources expander if sources are available."""
    if not sources:
        return

    with st.expander("View Sources", expanded=False):
        for j, src in enumerate(sources):
            st.markdown(f"**Source {j+1}:** {src['title']} ({src['source']})")
            st.markdown(f"Relevance: {src['score']:.4f}")
            if 'url' in src and src['url']:
                st.markdown(f"[Read more]({src['url']})")
            st.divider()


def _handle_user_question(retriever: Optional[NewsRetriever],
                         llm: Optional[NewsLLM]) -> None:
    """Handle user question input and generate response."""
    user_question = st.chat_input("Ask a question about the news")

    if not user_question:
        return

    # Show user question
    st.chat_message("user").write(user_question)

    # Process and show answer
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        answer_placeholder.info("Searching for relevant articles...")

        answer, sources = _process_question(user_question, retriever, llm, answer_placeholder)

        # Display the answer
        answer_placeholder.write(answer)

    # Add to chat history
    st.session_state['chat_history'].append((user_question, answer, sources))


def _process_question(question: str,
                     retriever: Optional[NewsRetriever],
                     llm: Optional[NewsLLM],
                     placeholder) -> tuple:
    """
    Process a user question and generate an answer.

    Args:
        question: The user's question
        retriever: The news retriever
        llm: The LLM interface
        placeholder: Streamlit placeholder for status updates

    Returns:
        tuple: (answer, sources)
    """
    # Get settings from session state
    show_query_processing = st.session_state.get("news_show_query", False)
    results_count = st.session_state.get("news_results_count", 10)

    # Check if retriever is available
    if retriever is None:
        placeholder.error("Retriever is not available.")
        return "I couldn't access the news database.", []

    # Process query
    search_query = _preprocess_query_if_needed(question, show_query_processing)

    # Search for relevant articles
    search_results = retriever.search(search_query, top_k=results_count)

    if not search_results:
        placeholder.warning("No relevant articles found.")
        return "I couldn't find any relevant information in the news", []

    # Generate answer based on search results
    answer = _generate_answer(question, search_results, llm, placeholder)

    # Display detailed sources
    _display_detailed_sources(search_results)

    return answer, search_results


def _preprocess_query_if_needed(question: str, show_processing: bool) -> str:
    """
    Preprocess the query if enabled.

    Args:
        question: The original question
        show_processing: Whether to show query processing

    Returns:
        str: The processed query
    """
    if not show_processing:
        return question

    enhanced_query, original_query = preprocess_query(question)
    st.info(f"Query processing: '{original_query}' → '{enhanced_query}'")
    return enhanced_query


def _generate_answer(question: str,
                    search_results: list,
                    llm: Optional[NewsLLM],
                    placeholder) -> str:
    """
    Generate an answer based on search results.

    Args:
        question: The user's question
        search_results: The retrieved search results
        llm: The LLM interface
        placeholder: Streamlit placeholder for status updates

    Returns:
        str: The generated answer
    """
    if llm is None:
        placeholder.warning("LLM is not available. Showing article summaries instead.")
        return _generate_fallback_summary(search_results)

    placeholder.info("Generating answer based on retrieved articles...")
    return llm.query(question, search_results)


def _generate_fallback_summary(search_results: list) -> str:
    """
    Generate a fallback summary when LLM is not available.

    Args:
        search_results: The retrieved search results

    Returns:
        str: The generated summary
    """
    summaries = []
    for i, result in enumerate(search_results[:3]):
        title = result.get('title', 'Untitled')
        source = result.get('source', 'Unknown')
        content = result.get('content_preview', result.get('text', ''))
        summaries.append(
            f"**Article {i+1}:** {title} from {source}\n\n{content[:200]}..."
        )

    return "Here are the most relevant articles I found:\n\n" + "\n\n".join(summaries)


def _display_detailed_sources(search_results: list) -> None:
    """
    Display detailed sources in an expander.

    Args:
        search_results: The search results to display
    """
    with st.expander("View Sources", expanded=False):
        for result in search_results:
            _display_single_source(result)


def _display_single_source(result: dict) -> None:
    """
    Display a single source with relevance information.

    Args:
        result: The result to display
    """
    col1, col2 = st.columns([1, 4])

    with col1:
        # Score indicator
        score = result['score']
        st.progress(score)
        st.caption(f"Relevance: {score:.4f}")

    with col2:
        st.markdown(f"**{result['title']}**")
        st.caption(f"Source: {result['source']}")
        if 'url' in result and result['url']:
            st.markdown(f"[Read original article]({result['url']})")

    preview_text = result.get('content_preview', result.get('text', ''))
    st.markdown(preview_text[:300] + "...")
    st.divider()
