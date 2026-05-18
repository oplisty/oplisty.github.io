---
permalink: /blog/
title: "Blog"
layout: archive
author_profile: true
stylesheets:
  - /assets/css/home.css
---

<h1 class="main-heading">Blog</h1>

<p class="blog-intro">Notes, updates, and writing collected in one place.</p>

<div class="blog-search-panel">
  <label class="blog-search-panel__label" for="blog-search-input">Search posts</label>
  <div class="blog-search-panel__input-wrap">
    <input id="blog-search-input" class="blog-search-panel__input" type="search" placeholder="Search by title, excerpt, tag, or category" aria-label="Search blog posts">
    <button id="blog-search-clear" class="blog-search-panel__clear" type="button">Clear</button>
  </div>
  <p id="blog-search-status" class="blog-search-panel__status" aria-live="polite"></p>
</div>

{% assign posts = site.blog | sort: 'date' | reverse %}

{% if posts.size > 0 %}
<div id="blog-card-list" class="blog-card-list">
  {% for post in posts %}
    {% assign category_list = post.categories | default: post.category %}
    <article class="blog-card"
      data-blog-search="{{ post.title | default: '' | downcase }} {{ post.excerpt | strip_html | default: '' | downcase }} {% if category_list %}{% if category_list.first %}{% for category in category_list %}{{ category | strip | downcase }} {% endfor %}{% else %}{{ category_list | downcase }}{% endif %}{% endif %} {% if post.tags %}{% for tag in post.tags %}{{ tag | downcase }} {% endfor %}{% endif %}">
      {% if post.cover %}
        <a class="blog-card__cover-link" href="{{ post.url | relative_url }}">
          <img class="blog-card__cover" src="{{ post.cover | relative_url }}" alt="{{ post.title }} cover image">
        </a>
      {% endif %}
      <div class="blog-card__content">
        <div class="blog-card__meta-row">
          {% if post.date %}
            <span class="blog-card__date">{{ post.date | date: "%B %d, %Y" }}</span>
          {% endif %}
          {% if post.read_time %}
            <span class="blog-card__reading-time"><i class="fa fa-clock-o" aria-hidden="true"></i> {% assign page = post %}{% include read-time.html %}</span>
          {% endif %}
        </div>

        <h2 class="blog-card__title">
          <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
        </h2>

        {% if post.category or post.categories %}
          <div class="blog-card__taxonomies">
            {% if category_list %}
              {% unless category_list.first %}
                {% assign category_list = category_list | split: ',' %}
              {% endunless %}
              {% for category in category_list %}
                <span class="blog-card__chip blog-card__chip--category">{{ category | strip }}</span>
              {% endfor %}
            {% endif %}
          </div>
        {% endif %}

        {% if post.tags %}
          <div class="blog-card__taxonomies">
            {% for tag in post.tags %}
              <span class="blog-card__chip blog-card__chip--tag">#{{ tag }}</span>
            {% endfor %}
          </div>
        {% endif %}

        {% if post.excerpt %}
          <p class="blog-card__excerpt">{{ post.excerpt | strip_html | truncate: 180 }}</p>
        {% endif %}
        <a class="blog-card__link" href="{{ post.url | relative_url }}">Read more</a>
      </div>
    </article>
  {% endfor %}
</div>
<div id="blog-search-empty" class="blog-empty-state" hidden>
  <p>No posts match your search.</p>
</div>
<script src="{{ '/assets/js/blog-search.js' | relative_url }}"></script>
{% else %}
<div class="blog-empty-state">
  <p>No blog posts yet.</p>
</div>
{% endif %}
