





<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
  <link rel="dns-prefetch" href="https://assets-cdn.github.com">
  <link rel="dns-prefetch" href="https://avatars0.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars1.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars2.githubusercontent.com">
  <link rel="dns-prefetch" href="https://avatars3.githubusercontent.com">
  <link rel="dns-prefetch" href="https://github-cloud.s3.amazonaws.com">
  <link rel="dns-prefetch" href="https://user-images.githubusercontent.com/">



  <link crossorigin="anonymous" media="all" integrity="sha512-hqbuBb0QOOmiWgl8a1V1N5q6TI/G0A2hVt/lCFYafR+fYsuXeRUcsdcb/yUyVEHYXktmUXl0Mx9s/BOUNZVq4w==" rel="stylesheet" href="https://assets-cdn.github.com/assets/frameworks-23c9e7262eee71bc6f67f6950190a162.css" />
  <link crossorigin="anonymous" media="all" integrity="sha512-3hVGKwp3EP1XYe7jtewDAU5o/gNueZ3Xm7TL5zKnDaYFi8IeOCqIGitZS1FxnwJp75FLsc7VZSVUbez2mbopJg==" rel="stylesheet" href="https://assets-cdn.github.com/assets/github-9c3638342476c515de9c43a886aa2754.css" />
  
  
  
  

  <meta name="viewport" content="width=device-width">
  
  <title>mosesdecoder/multi-bleu.perl at master · moses-smt/mosesdecoder</title>
    <meta name="description" content="GitHub is where people build software. More than 27 million people use GitHub to discover, fork, and contribute to over 80 million projects.">
  <link rel="search" type="application/opensearchdescription+xml" href="/opensearch.xml" title="GitHub">
  <link rel="fluid-icon" href="https://github.com/fluidicon.png" title="GitHub">
  <meta property="fb:app_id" content="1401488693436528">

    
    <meta property="og:image" content="https://avatars2.githubusercontent.com/u/1038770?s=400&amp;v=4" /><meta property="og:site_name" content="GitHub" /><meta property="og:type" content="object" /><meta property="og:title" content="moses-smt/mosesdecoder" /><meta property="og:url" content="https://github.com/moses-smt/mosesdecoder" /><meta property="og:description" content="mosesdecoder - Moses, the machine translation system" />

  <link rel="assets" href="https://assets-cdn.github.com/">
  <link rel="web-socket" href="wss://live.github.com/_sockets/VjI6Mjc2OTU3NjM5OmE5ZWY5YWY2ZDM2NWQ4Y2Y5YmVjOGVkYTI3NjBiNjZmNjJhZTllZmZjNTA0NjQzZmM1MTU0ZWExZDJjOGVmODg=--1b33a3e3922d5e2e8c2515c6cbfcae13f629778d">
  <meta name="pjax-timeout" content="1000">
  <link rel="sudo-modal" href="/sessions/sudo_modal">
  <meta name="request-id" content="DC8A:3683:20D2EEA:3D5EB8A:5B03EB56" data-pjax-transient>


  

  <meta name="selected-link" value="repo_source" data-pjax-transient>

    <meta name="google-site-verification" content="KT5gs8h0wvaagLKAVWq8bbeNwnZZK1r1XQysX3xurLU">
  <meta name="google-site-verification" content="ZzhVyEFwb7w3e0-uOTltm8Jsck2F5StVihD0exw2fsA">
  <meta name="google-site-verification" content="GXs5KoUUkNCoaAZn7wPN-t01Pywp9M3sEjnt_3_ZWPc">
    <meta name="google-analytics" content="UA-3769691-2">

<meta name="octolytics-host" content="collector.githubapp.com" /><meta name="octolytics-app-id" content="github" /><meta name="octolytics-event-url" content="https://collector.githubapp.com/github-external/browser_event" /><meta name="octolytics-dimension-request_id" content="DC8A:3683:20D2EEA:3D5EB8A:5B03EB56" /><meta name="octolytics-dimension-region_edge" content="iad" /><meta name="octolytics-dimension-region_render" content="iad" /><meta name="octolytics-actor-id" content="24842609" /><meta name="octolytics-actor-login" content="simonevanbruggen" /><meta name="octolytics-actor-hash" content="1f7aec8cd6ab11b9177218881b3e7425dee2e23ad23fd3c26b60ead04debc2ae" />
<meta name="analytics-location" content="/&lt;user-name&gt;/&lt;repo-name&gt;/blob/show" data-pjax-transient="true" />




  <meta class="js-ga-set" name="dimension1" content="Logged In">


  

      <meta name="hostname" content="github.com">
    <meta name="user-login" content="simonevanbruggen">

      <meta name="expected-hostname" content="github.com">
    <meta name="js-proxy-site-detection-payload" content="MGJjMjJmNDE5YmUyZDY5ZGM2MzYwOTIzMjcwYjYyM2U0YzUzYWNlNTJmOGMzMzgyM2Y2YzVkZGU3NzgxYmFhM3x7InJlbW90ZV9hZGRyZXNzIjoiMTQ1LjE1LjI0NC4xNyIsInJlcXVlc3RfaWQiOiJEQzhBOjM2ODM6MjBEMkVFQTozRDVFQjhBOjVCMDNFQjU2IiwidGltZXN0YW1wIjoxNTI2OTgzNTE0LCJob3N0IjoiZ2l0aHViLmNvbSJ9">

    <meta name="enabled-features" content="ASYNC_NEWS_FEED,UNIVERSE_BANNER,FREE_TRIALS,MARKETPLACE_INSIGHTS,MARKETPLACE_SELF_SERVE,MARKETPLACE_INSIGHTS_CONVERSION_PERCENTAGES,NEWS_FEED_EVENT_ROLLUPS,STAR_FROM_NEWS_FEED,FOLLOW_FROM_NEWS_FEED">

  <meta name="html-safe-nonce" content="e2d64cb603f079dea876edd7c0acae7f9f1786bc">

  <meta http-equiv="x-pjax-version" content="43224e8c105deecf285e4bd9cde870aa">
  

      <link href="https://github.com/moses-smt/mosesdecoder/commits/master.atom" rel="alternate" title="Recent Commits to mosesdecoder:master" type="application/atom+xml">

  <meta name="description" content="mosesdecoder - Moses, the machine translation system">
  <meta name="go-import" content="github.com/moses-smt/mosesdecoder git https://github.com/moses-smt/mosesdecoder.git">

  <meta name="octolytics-dimension-user_id" content="1038770" /><meta name="octolytics-dimension-user_login" content="moses-smt" /><meta name="octolytics-dimension-repository_id" content="2236146" /><meta name="octolytics-dimension-repository_nwo" content="moses-smt/mosesdecoder" /><meta name="octolytics-dimension-repository_public" content="true" /><meta name="octolytics-dimension-repository_is_fork" content="false" /><meta name="octolytics-dimension-repository_network_root_id" content="2236146" /><meta name="octolytics-dimension-repository_network_root_nwo" content="moses-smt/mosesdecoder" /><meta name="octolytics-dimension-repository_explore_github_marketplace_ci_cta_shown" content="false" />


    <link rel="canonical" href="https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl" data-pjax-transient>


  <meta name="browser-stats-url" content="https://api.github.com/_private/browser/stats">

  <meta name="browser-errors-url" content="https://api.github.com/_private/browser/errors">

  <link rel="mask-icon" href="https://assets-cdn.github.com/pinned-octocat.svg" color="#000000">
  <link rel="icon" type="image/x-icon" class="js-site-favicon" href="https://assets-cdn.github.com/favicon.ico">

<meta name="theme-color" content="#1e2327">


  <meta name="u2f-support" content="true">

<link rel="manifest" href="/manifest.json" crossOrigin="use-credentials">

  </head>

  <body class="logged-in env-production page-blob">
    

  <div class="position-relative js-header-wrapper ">
    <a href="#start-of-content" tabindex="1" class="p-3 bg-blue text-white show-on-focus js-skip-to-content">Skip to content</a>
    <div id="js-pjax-loader-bar" class="pjax-loader-bar"><div class="progress"></div></div>

    
    
    



        
<header class="Header  f5" role="banner">
  <div class="d-flex flex-justify-between px-3 container-lg">
    <div class="d-flex flex-justify-between ">
      <div class="">
        <a class="header-logo-invertocat" href="https://github.com/" data-hotkey="g d" aria-label="Homepage" data-ga-click="Header, go to dashboard, icon:logo">
  <svg height="32" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="32" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>

      </div>

    </div>

    <div class="HeaderMenu d-flex flex-justify-between flex-auto">
      <div class="d-flex">
            <div class="">
              <div class="header-search scoped-search site-scoped-search js-site-search" role="search">
  <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-site-search-form" data-scope-type="Repository" data-scope-id="2236146" data-scoped-search-url="/moses-smt/mosesdecoder/search" data-unscoped-search-url="/search" action="/moses-smt/mosesdecoder/search" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" />
    <label class="form-control header-search-wrapper  js-chromeless-input-container">
          <a class="header-search-scope no-underline" href="/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl">This repository</a>
      <input type="text"
        class="form-control header-search-input  js-site-search-focus js-site-search-field is-clearable"
        data-hotkey="s,/"
        name="q"
        value=""
        placeholder="Search"
        aria-label="Search this repository"
        data-unscoped-placeholder="Search GitHub"
        data-scoped-placeholder="Search"
        autocapitalize="off"
        >
        <input type="hidden" class="js-site-search-type-field" name="type" >
    </label>
</form></div>

            </div>

          <ul class="d-flex pl-2 flex-items-center text-bold list-style-none" role="navigation">
            <li>
              <a class="js-selected-navigation-item HeaderNavlink px-2" data-hotkey="g p" data-ga-click="Header, click, Nav menu - item:pulls context:user" aria-label="Pull requests you created" data-selected-links="/pulls /pulls/assigned /pulls/mentioned /pulls" href="/pulls">
                Pull requests
</a>            </li>
            <li>
              <a class="js-selected-navigation-item HeaderNavlink px-2" data-hotkey="g i" data-ga-click="Header, click, Nav menu - item:issues context:user" aria-label="Issues you created" data-selected-links="/issues /issues/assigned /issues/mentioned /issues" href="/issues">
                Issues
</a>            </li>
                <li>
                  <a class="js-selected-navigation-item HeaderNavlink px-2" data-ga-click="Header, click, Nav menu - item:marketplace context:user" data-octo-click="marketplace_click" data-octo-dimensions="location:nav_bar, group:" data-selected-links=" /marketplace" href="/marketplace">
                    Marketplace
</a>                </li>
            <li>
              <a class="js-selected-navigation-item HeaderNavlink px-2" data-ga-click="Header, click, Nav menu - item:explore" data-selected-links="/explore /trending /trending/developers /integrations /integrations/feature/code /integrations/feature/collaborate /integrations/feature/ship showcases showcases_search showcases_landing /explore" href="/explore">
                Explore
</a>            </li>
          </ul>
      </div>

      <div class="d-flex">
        
<ul class="user-nav d-flex flex-items-center list-style-none" id="user-links">
  <li class="dropdown js-menu-container">
    <span class="d-inline-block  px-2">
      

    </span>
  </li>

  <li class="dropdown js-menu-container">
    <details class="details-expanded details-reset js-dropdown-details d-flex px-2 flex-items-center">
      <summary class="HeaderNavlink"
         aria-label="Create new…"
         data-ga-click="Header, create new, icon:add">
        <svg class="octicon octicon-plus float-left mr-1 mt-1" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 9H7v5H5V9H0V7h5V2h2v5h5v2z"/></svg>
        <span class="dropdown-caret mt-1"></span>
      </summary>

      <ul class="dropdown-menu dropdown-menu-sw">
        
<a class="dropdown-item" href="/new" data-ga-click="Header, create new repository">
  New repository
</a>

  <a class="dropdown-item" href="/new/import" data-ga-click="Header, import a repository">
    Import repository
  </a>

<a class="dropdown-item" href="https://gist.github.com/" data-ga-click="Header, create new gist">
  New gist
</a>

  <a class="dropdown-item" href="/organizations/new" data-ga-click="Header, create new organization">
    New organization
  </a>



  <div class="dropdown-divider"></div>
  <div class="dropdown-header">
    <span title="moses-smt/mosesdecoder">This repository</span>
  </div>
    <a class="dropdown-item" href="/moses-smt/mosesdecoder/issues/new" data-ga-click="Header, create new issue">
      New issue
    </a>

      </ul>
    </details>
  </li>

  <li class="dropdown js-menu-container">

    <details class="details-expanded details-reset js-dropdown-details d-flex pl-2 flex-items-center">
      <summary class="HeaderNavlink name mt-1"
        aria-label="View profile and more"
        data-ga-click="Header, show menu, icon:avatar">
        <img alt="@simonevanbruggen" class="avatar float-left mr-1" src="https://avatars0.githubusercontent.com/u/24842609?s=40&amp;v=4" height="20" width="20">
        <span class="dropdown-caret"></span>
      </summary>

      <ul class="dropdown-menu dropdown-menu-sw">
        <li class="dropdown-header header-nav-current-user css-truncate">
          Signed in as <strong class="css-truncate-target">simonevanbruggen</strong>
        </li>

        <li class="dropdown-divider"></li>

        <li><a class="dropdown-item" href="/simonevanbruggen" data-ga-click="Header, go to profile, text:your profile">
          Your profile
        </a></li>
        <li><a class="dropdown-item" href="/simonevanbruggen?tab=stars" data-ga-click="Header, go to starred repos, text:your stars">
          Your stars
        </a></li>
          <li><a class="dropdown-item" href="https://gist.github.com/" data-ga-click="Header, your gists, text:your gists">Your gists</a></li>

        <li class="dropdown-divider"></li>

        <li><a class="dropdown-item" href="https://help.github.com" data-ga-click="Header, go to help, text:help">
          Help
        </a></li>

        <li><a class="dropdown-item" href="/settings/profile" data-ga-click="Header, go to settings, icon:settings">
          Settings
        </a></li>

        <li><!-- '"` --><!-- </textarea></xmp> --></option></form><form class="logout-form" action="/logout" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="rKKbUBHPiRcdr0olTVZbZRHD0wgKEiUHA+/mU25BhZpdc1FZFUUgPiSoe1D8EbUDhhEbxb923bTFS+5gDTmB4Q==" />
          <button type="submit" class="dropdown-item dropdown-signout" data-ga-click="Header, sign out, icon:logout">
            Sign out
          </button>
        </form></li>
      </ul>
    </details>
  </li>
</ul>



        <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="sr-only right-0" action="/logout" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="uNRWqTroxuEQkSVrra0YCeZK3ZNxXHW8eKPRCU+MdOVJBZygPmJvyCmWFB4c6vZvcZgVXsQ4jQ++B9k6LPRwng==" />
          <button type="submit" class="dropdown-item dropdown-signout" data-ga-click="Header, sign out, icon:logout">
            Sign out
          </button>
</form>      </div>
    </div>
  </div>
</header>

      

  </div>

  <div id="start-of-content" class="show-on-focus"></div>

    <div id="js-flash-container">
</div>



  <div role="main" class="application-main ">
        <div itemscope itemtype="http://schema.org/SoftwareSourceCode" class="">
    <div id="js-repo-pjax-container" data-pjax-container >
      





  



  <div class="pagehead repohead instapaper_ignore readability-menu experiment-repo-nav  ">
    <div class="repohead-details-container clearfix container">

      <ul class="pagehead-actions">
  <li>
        <!-- '"` --><!-- </textarea></xmp> --></option></form><form data-autosubmit="true" data-remote="true" class="js-social-container" action="/notifications/subscribe" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="3wyqYyFp+MG43Z3AY4sYWHSnIHvx8ygy2Ho+MthpjiCb/1ZISitG53LtItl3bwdzTnNe4a9otw8U3NNRxcTRug==" />      <input type="hidden" name="repository_id" id="repository_id" value="2236146" class="form-control" />

        <div class="select-menu js-menu-container js-select-menu">
          <a href="/moses-smt/mosesdecoder/subscription"
            class="btn btn-sm btn-with-count select-menu-button js-menu-target"
            role="button"
            aria-haspopup="true"
            aria-expanded="false"
            aria-label="Toggle repository notifications menu"
            data-ga-click="Repository, click Watch settings, action:blob#show">
            <span class="js-select-button">
                <svg class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                Watch
            </span>
          </a>
          <a class="social-count js-social-count"
            href="/moses-smt/mosesdecoder/watchers"
            aria-label="154 users are watching this repository">
            154
          </a>

        <div class="select-menu-modal-holder">
          <div class="select-menu-modal subscription-menu-modal js-menu-content">
            <div class="select-menu-header js-navigation-enable" tabindex="-1">
              <svg class="octicon octicon-x js-menu-close" role="img" aria-label="Close" viewBox="0 0 12 16" version="1.1" width="12" height="16"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
              <span class="select-menu-title">Notifications</span>
            </div>

              <div class="select-menu-list js-navigation-container" role="menu">

                <div class="select-menu-item js-navigation-item selected" role="menuitem" tabindex="0">
                  <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
                  <div class="select-menu-item-text">
                    <input type="radio" name="do" id="do_included" value="included" checked="checked" />
                    <span class="select-menu-item-heading">Not watching</span>
                    <span class="description">Be notified when participating or @mentioned.</span>
                    <span class="js-select-button-text hidden-select-button-text">
                      <svg class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                      Watch
                    </span>
                  </div>
                </div>

                <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
                  <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
                  <div class="select-menu-item-text">
                    <input type="radio" name="do" id="do_subscribed" value="subscribed" />
                    <span class="select-menu-item-heading">Watching</span>
                    <span class="description">Be notified of all conversations.</span>
                    <span class="js-select-button-text hidden-select-button-text">
                      <svg class="octicon octicon-eye" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.06 2C3 2 0 8 0 8s3 6 8.06 6C13 14 16 8 16 8s-3-6-7.94-6zM8 12c-2.2 0-4-1.78-4-4 0-2.2 1.8-4 4-4 2.22 0 4 1.8 4 4 0 2.22-1.78 4-4 4zm2-4c0 1.11-.89 2-2 2-1.11 0-2-.89-2-2 0-1.11.89-2 2-2 1.11 0 2 .89 2 2z"/></svg>
                        Unwatch
                    </span>
                  </div>
                </div>

                <div class="select-menu-item js-navigation-item " role="menuitem" tabindex="0">
                  <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
                  <div class="select-menu-item-text">
                    <input type="radio" name="do" id="do_ignore" value="ignore" />
                    <span class="select-menu-item-heading">Ignoring</span>
                    <span class="description">Never be notified.</span>
                    <span class="js-select-button-text hidden-select-button-text">
                      <svg class="octicon octicon-mute" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 2.81v10.38c0 .67-.81 1-1.28.53L3 10H1c-.55 0-1-.45-1-1V7c0-.55.45-1 1-1h2l3.72-3.72C7.19 1.81 8 2.14 8 2.81zm7.53 3.22l-1.06-1.06-1.97 1.97-1.97-1.97-1.06 1.06L11.44 8 9.47 9.97l1.06 1.06 1.97-1.97 1.97 1.97 1.06-1.06L13.56 8l1.97-1.97z"/></svg>
                        Stop ignoring
                    </span>
                  </div>
                </div>

              </div>

            </div>
          </div>
        </div>
</form>
  </li>

  <li>
    
  <div class="js-toggler-container js-social-container starring-container ">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="starred js-social-form" action="/moses-smt/mosesdecoder/unstar" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="9S/4EVlH/bYo4Md7a+dE9VdMMrExD8CdHsAL4XFTv4ZUucvSwusxKPPk680OKrbCQTLJhHbM5sVPbAZaJZW/4w==" />
      <input type="hidden" name="context" value="repository"></input>
      <button
        type="submit"
        class="btn btn-sm btn-with-count js-toggler-target"
        aria-label="Unstar this repository" title="Unstar moses-smt/mosesdecoder"
        data-ga-click="Repository, click unstar button, action:blob#show; text:Unstar">
        <svg class="octicon octicon-star" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74L14 6z"/></svg>
        Unstar
      </button>
        <a class="social-count js-social-count" href="/moses-smt/mosesdecoder/stargazers"
           aria-label="740 users starred this repository">
          740
        </a>
</form>
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="unstarred js-social-form" action="/moses-smt/mosesdecoder/star" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="M80mjVya6a0b7HmGMuQsG0gAJbkzkAvbxxEO22/xsS6FcdE9osaFAWOndtPnOQf1UpL/uHphcfcLK+yVXLtglg==" />
      <input type="hidden" name="context" value="repository"></input>
      <button
        type="submit"
        class="btn btn-sm btn-with-count js-toggler-target"
        aria-label="Star this repository" title="Star moses-smt/mosesdecoder"
        data-ga-click="Repository, click star button, action:blob#show; text:Star">
        <svg class="octicon octicon-star" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M14 6l-4.9-.64L7 1 4.9 5.36 0 6l3.6 3.26L2.67 14 7 11.67 11.33 14l-.93-4.74L14 6z"/></svg>
        Star
      </button>
        <a class="social-count js-social-count" href="/moses-smt/mosesdecoder/stargazers"
           aria-label="740 users starred this repository">
          740
        </a>
</form>  </div>

  </li>

  <li>
          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="btn-with-count" action="/moses-smt/mosesdecoder/fork" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="iRSdk0feWFfkFt2Wa2E59JOcDWLY59Wpu5baDugezyz7Nscx6nApjnbYqFKqd12M03nx5k00Pq7an8JNURDq8Q==" />
            <button
                type="submit"
                class="btn btn-sm btn-with-count"
                data-ga-click="Repository, show fork modal, action:blob#show; text:Fork"
                title="Fork your own copy of moses-smt/mosesdecoder to your account"
                aria-label="Fork your own copy of moses-smt/mosesdecoder to your account">
              <svg class="octicon octicon-repo-forked" viewBox="0 0 10 16" version="1.1" width="10" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8 1a1.993 1.993 0 0 0-1 3.72V6L5 8 3 6V4.72A1.993 1.993 0 0 0 2 1a1.993 1.993 0 0 0-1 3.72V6.5l3 3v1.78A1.993 1.993 0 0 0 5 15a1.993 1.993 0 0 0 1-3.72V9.5l3-3V4.72A1.993 1.993 0 0 0 8 1zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3 10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zm3-10c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
              Fork
            </button>
</form>
    <a href="/moses-smt/mosesdecoder/network" class="social-count"
       aria-label="460 users forked this repository">
      460
    </a>
  </li>
</ul>

      <h1 class="public ">
  <svg class="octicon octicon-repo" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M4 9H3V8h1v1zm0-3H3v1h1V6zm0-2H3v1h1V4zm0-2H3v1h1V2zm8-1v12c0 .55-.45 1-1 1H6v2l-1.5-1.5L3 16v-2H1c-.55 0-1-.45-1-1V1c0-.55.45-1 1-1h10c.55 0 1 .45 1 1zm-1 10H1v2h2v-1h3v1h5v-2zm0-10H2v9h9V1z"/></svg>
  <span class="author" itemprop="author"><a class="url fn" rel="author" href="/moses-smt">moses-smt</a></span><!--
--><span class="path-divider">/</span><!--
--><strong itemprop="name"><a data-pjax="#js-repo-pjax-container" href="/moses-smt/mosesdecoder">mosesdecoder</a></strong>

</h1>

    </div>
    
<nav class="reponav js-repo-nav js-sidenav-container-pjax container"
     itemscope
     itemtype="http://schema.org/BreadcrumbList"
     role="navigation"
     data-pjax="#js-repo-pjax-container">

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a class="js-selected-navigation-item selected reponav-item" itemprop="url" data-hotkey="g c" data-selected-links="repo_source repo_downloads repo_commits repo_releases repo_tags repo_branches repo_packages /moses-smt/mosesdecoder" href="/moses-smt/mosesdecoder">
      <svg class="octicon octicon-code" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M9.5 3L8 4.5 11.5 8 8 11.5 9.5 13 14 8 9.5 3zm-5 0L0 8l4.5 5L6 11.5 2.5 8 6 4.5 4.5 3z"/></svg>
      <span itemprop="name">Code</span>
      <meta itemprop="position" content="1">
</a>  </span>

    <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
      <a itemprop="url" data-hotkey="g i" class="js-selected-navigation-item reponav-item" data-selected-links="repo_issues repo_labels repo_milestones /moses-smt/mosesdecoder/issues" href="/moses-smt/mosesdecoder/issues">
        <svg class="octicon octicon-issue-opened" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"/></svg>
        <span itemprop="name">Issues</span>
        <span class="Counter">2</span>
        <meta itemprop="position" content="2">
</a>    </span>

  <span itemscope itemtype="http://schema.org/ListItem" itemprop="itemListElement">
    <a data-hotkey="g p" itemprop="url" class="js-selected-navigation-item reponav-item" data-selected-links="repo_pulls checks /moses-smt/mosesdecoder/pulls" href="/moses-smt/mosesdecoder/pulls">
      <svg class="octicon octicon-git-pull-request" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11 11.28V5c-.03-.78-.34-1.47-.94-2.06C9.46 2.35 8.78 2.03 8 2H7V0L4 3l3 3V4h1c.27.02.48.11.69.31.21.2.3.42.31.69v6.28A1.993 1.993 0 0 0 10 15a1.993 1.993 0 0 0 1-3.72zm-1 2.92c-.66 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2zM4 3c0-1.11-.89-2-2-2a1.993 1.993 0 0 0-1 3.72v6.56A1.993 1.993 0 0 0 2 15a1.993 1.993 0 0 0 1-3.72V4.72c.59-.34 1-.98 1-1.72zm-.8 10c0 .66-.55 1.2-1.2 1.2-.65 0-1.2-.55-1.2-1.2 0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2zM2 4.2C1.34 4.2.8 3.65.8 3c0-.65.55-1.2 1.2-1.2.65 0 1.2.55 1.2 1.2 0 .65-.55 1.2-1.2 1.2z"/></svg>
      <span itemprop="name">Pull requests</span>
      <span class="Counter">1</span>
      <meta itemprop="position" content="3">
</a>  </span>

    <a data-hotkey="g b" class="js-selected-navigation-item reponav-item" data-selected-links="repo_projects new_repo_project repo_project /moses-smt/mosesdecoder/projects" href="/moses-smt/mosesdecoder/projects">
      <svg class="octicon octicon-project" viewBox="0 0 15 16" version="1.1" width="15" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M10 12h3V2h-3v10zm-4-2h3V2H6v8zm-4 4h3V2H2v12zm-1 1h13V1H1v14zM14 0H1a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h13a1 1 0 0 0 1-1V1a1 1 0 0 0-1-1z"/></svg>
      Projects
      <span class="Counter" >0</span>
</a>


  <a class="js-selected-navigation-item reponav-item" data-selected-links="repo_graphs repo_contributors dependency_graph pulse /moses-smt/mosesdecoder/pulse" href="/moses-smt/mosesdecoder/pulse">
    <svg class="octicon octicon-graph" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M16 14v1H0V0h1v14h15zM5 13H3V8h2v5zm4 0H7V3h2v10zm4 0h-2V6h2v7z"/></svg>
    Insights
</a>

</nav>


  </div>

<div class="container new-discussion-timeline experiment-repo-nav  ">
  <div class="repository-content ">

    
  <a class="d-none js-permalink-shortcut" data-hotkey="y" href="/moses-smt/mosesdecoder/blob/b56d97dee0e9e452d2ffa08e809a90917ad7e7a7/scripts/generic/multi-bleu.perl">Permalink</a>

  <!-- blob contrib key: blob_contributors:v21:509699a3051c87f4e49168b9bc234b06 -->

  <div class="file-navigation">
    
<div class="select-menu branch-select-menu js-menu-container js-select-menu float-left">
  <button class=" btn btn-sm select-menu-button js-menu-target css-truncate" data-hotkey="w"
    
    type="button" aria-label="Switch branches or tags" aria-expanded="false" aria-haspopup="true">
      <i>Branch:</i>
      <span class="js-select-button css-truncate-target">master</span>
  </button>

  <div class="select-menu-modal-holder js-menu-content js-navigation-container" data-pjax>

    <div class="select-menu-modal">
      <div class="select-menu-header">
        <svg class="octicon octicon-x js-menu-close" role="img" aria-label="Close" viewBox="0 0 12 16" version="1.1" width="12" height="16"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
        <span class="select-menu-title">Switch branches/tags</span>
      </div>

      <div class="select-menu-filters">
        <div class="select-menu-text-filter">
          <input type="text" aria-label="Filter branches/tags" id="context-commitish-filter-field" class="form-control js-filterable-field js-navigation-enable" placeholder="Filter branches/tags">
        </div>
        <div class="select-menu-tabs">
          <ul>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="branches" data-filter-placeholder="Filter branches/tags" class="js-select-menu-tab" role="tab">Branches</a>
            </li>
            <li class="select-menu-tab">
              <a href="#" data-tab-filter="tags" data-filter-placeholder="Find a tag…" class="js-select-menu-tab" role="tab">Tags</a>
            </li>
          </ul>
        </div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="branches" role="menu">

        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/BEFORE_FF_FRAMEWORK/scripts/generic/multi-bleu.perl"
               data-name="BEFORE_FF_FRAMEWORK"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                BEFORE_FF_FRAMEWORK
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/CheckTargetNgrams/scripts/generic/multi-bleu.perl"
               data-name="CheckTargetNgrams"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                CheckTargetNgrams
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/CompactPT/scripts/generic/multi-bleu.perl"
               data-name="CompactPT"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                CompactPT
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/DPR_MOSES/scripts/generic/multi-bleu.perl"
               data-name="DPR_MOSES"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                DPR_MOSES
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/FF_ttptr/scripts/generic/multi-bleu.perl"
               data-name="FF_ttptr"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                FF_ttptr
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/GHK/scripts/generic/multi-bleu.perl"
               data-name="GHK"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                GHK
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/GHKMStruct/scripts/generic/multi-bleu.perl"
               data-name="GHKMStruct"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                GHKMStruct
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/QE/scripts/generic/multi-bleu.perl"
               data-name="QE"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                QE
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-0.91/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-0.91"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-0.91
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-1.0/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-1.0"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-1.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-2.1/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-2.1"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-2.1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-2.1.1/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-2.1.1"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-2.1.1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-3.0-cpp14/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-3.0-cpp14"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-3.0-cpp14
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-3.0/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-3.0"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-3.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-4.0-RC1/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-4.0-RC1"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-4.0-RC1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-4.0-RC2/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-4.0-RC2"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-4.0-RC2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-4.0-RC3/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-4.0-RC3"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-4.0-RC3
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/RELEASE-4.0/scripts/generic/multi-bleu.perl"
               data-name="RELEASE-4.0"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                RELEASE-4.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/abhi_discr/scripts/generic/multi-bleu.perl"
               data-name="abhi_discr"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                abhi_discr
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/alvations-patch-1/scripts/generic/multi-bleu.perl"
               data-name="alvations-patch-1"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                alvations-patch-1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/alvations-python-tokenizer/scripts/generic/multi-bleu.perl"
               data-name="alvations-python-tokenizer"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                alvations-python-tokenizer
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/amittai/scripts/generic/multi-bleu.perl"
               data-name="amittai"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                amittai
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/andre_lynum/scripts/generic/multi-bleu.perl"
               data-name="andre_lynum"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                andre_lynum
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/android-ui/scripts/generic/multi-bleu.perl"
               data-name="android-ui"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                android-ui
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/apply-coding-standard/scripts/generic/multi-bleu.perl"
               data-name="apply-coding-standard"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                apply-coding-standard
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/async-factors/scripts/generic/multi-bleu.perl"
               data-name="async-factors"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                async-factors
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/augmLMResult/scripts/generic/multi-bleu.perl"
               data-name="augmLMResult"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                augmLMResult
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/auli_constraint/scripts/generic/multi-bleu.perl"
               data-name="auli_constraint"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                auli_constraint
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/bertoldi_onlinecommand/scripts/generic/multi-bleu.perl"
               data-name="bertoldi_onlinecommand"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bertoldi_onlinecommand
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/biased-bitext-sampling/scripts/generic/multi-bleu.perl"
               data-name="biased-bitext-sampling"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                biased-bitext-sampling
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/bilingual-lm-paul/scripts/generic/multi-bleu.perl"
               data-name="bilingual-lm-paul"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bilingual-lm-paul
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/bilingual-lm/scripts/generic/multi-bleu.perl"
               data-name="bilingual-lm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                bilingual-lm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/braune_longdist/scripts/generic/multi-bleu.perl"
               data-name="braune_longdist"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                braune_longdist
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/cdyer-multifactor/scripts/generic/multi-bleu.perl"
               data-name="cdyer-multifactor"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                cdyer-multifactor
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/chardmeier-coref/scripts/generic/multi-bleu.perl"
               data-name="chardmeier-coref"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                chardmeier-coref
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/chardmeier-ldc/scripts/generic/multi-bleu.perl"
               data-name="chardmeier-ldc"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                chardmeier-ldc
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/chardmeier-lminterp/scripts/generic/multi-bleu.perl"
               data-name="chardmeier-lminterp"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                chardmeier-lminterp
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/chart_alignments/scripts/generic/multi-bleu.perl"
               data-name="chart_alignments"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                chart_alignments
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/conditional-oxlm/scripts/generic/multi-bleu.perl"
               data-name="conditional-oxlm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                conditional-oxlm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/config-switching/scripts/generic/multi-bleu.perl"
               data-name="config-switching"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                config-switching
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/dalsa/scripts/generic/multi-bleu.perl"
               data-name="dalsa"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                dalsa
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/damt-phrase2master/scripts/generic/multi-bleu.perl"
               data-name="damt-phrase2master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                damt-phrase2master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/damt_common/scripts/generic/multi-bleu.perl"
               data-name="damt_common"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                damt_common
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/damt_hiero/scripts/generic/multi-bleu.perl"
               data-name="damt_hiero"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                damt_hiero
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/damt_phrase/scripts/generic/multi-bleu.perl"
               data-name="damt_phrase"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                damt_phrase
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/dumb_psd/scripts/generic/multi-bleu.perl"
               data-name="dumb_psd"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                dumb_psd
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/dynamic-models/scripts/generic/multi-bleu.perl"
               data-name="dynamic-models"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                dynamic-models
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/dynamic-phrase-tables/scripts/generic/multi-bleu.perl"
               data-name="dynamic-phrase-tables"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                dynamic-phrase-tables
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/dynsapt/scripts/generic/multi-bleu.perl"
               data-name="dynsapt"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                dynsapt
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/eva_cube_pruning/scripts/generic/multi-bleu.perl"
               data-name="eva_cube_pruning"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                eva_cube_pruning
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/eva_maxent/scripts/generic/multi-bleu.perl"
               data-name="eva_maxent"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                eva_maxent
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/factored_hacks/scripts/generic/multi-bleu.perl"
               data-name="factored_hacks"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                factored_hacks
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/fethi/scripts/generic/multi-bleu.perl"
               data-name="fethi"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                fethi
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/fix-wchar-subscripts/scripts/generic/multi-bleu.perl"
               data-name="fix-wchar-subscripts"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                fix-wchar-subscripts
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/fscorer/scripts/generic/multi-bleu.perl"
               data-name="fscorer"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                fscorer
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/grace_cube_pruning_gen/scripts/generic/multi-bleu.perl"
               data-name="grace_cube_pruning_gen"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                grace_cube_pruning_gen
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/hierarchical-reo/scripts/generic/multi-bleu.perl"
               data-name="hierarchical-reo"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                hierarchical-reo
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/hieu_ftemplate_2/scripts/generic/multi-bleu.perl"
               data-name="hieu_ftemplate_2"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                hieu_ftemplate_2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/hieu_iphone/scripts/generic/multi-bleu.perl"
               data-name="hieu_iphone"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                hieu_iphone
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/jjmoses/scripts/generic/multi-bleu.perl"
               data-name="jjmoses"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jjmoses
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/jointlm_improvements/scripts/generic/multi-bleu.perl"
               data-name="jointlm_improvements"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jointlm_improvements
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/josiah/scripts/generic/multi-bleu.perl"
               data-name="josiah"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                josiah
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/jozef_pipeline/scripts/generic/multi-bleu.perl"
               data-name="jozef_pipeline"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                jozef_pipeline
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/kenlm-prefetch/scripts/generic/multi-bleu.perl"
               data-name="kenlm-prefetch"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                kenlm-prefetch
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/lane-constraint/scripts/generic/multi-bleu.perl"
               data-name="lane-constraint"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lane-constraint
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/lane-mtm3/scripts/generic/multi-bleu.perl"
               data-name="lane-mtm3"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lane-mtm3
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/lane-multi/scripts/generic/multi-bleu.perl"
               data-name="lane-multi"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lane-multi
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/lane-syntax/scripts/generic/multi-bleu.perl"
               data-name="lane-syntax"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lane-syntax
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/lane_moses2/scripts/generic/multi-bleu.perl"
               data-name="lane_moses2"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lane_moses2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/lexi-smoothing/scripts/generic/multi-bleu.perl"
               data-name="lexi-smoothing"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lexi-smoothing
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/lexi_RELEASE-1.0/scripts/generic/multi-bleu.perl"
               data-name="lexi_RELEASE-1.0"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                lexi_RELEASE-1.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/local_lms/scripts/generic/multi-bleu.perl"
               data-name="local_lms"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                local_lms
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/maria_HeadDrivenFeature/scripts/generic/multi-bleu.perl"
               data-name="maria_HeadDrivenFeature"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                maria_HeadDrivenFeature
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/maria_MTM14/scripts/generic/multi-bleu.perl"
               data-name="maria_MTM14"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                maria_MTM14
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/maria_SelPref/scripts/generic/multi-bleu.perl"
               data-name="maria_SelPref"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                maria_SelPref
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/maria_SyntaxFeatures/scripts/generic/multi-bleu.perl"
               data-name="maria_SyntaxFeatures"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                maria_SyntaxFeatures
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open selected"
               href="/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl"
               data-name="master"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                master
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/matecat_async/scripts/generic/multi-bleu.perl"
               data-name="matecat_async"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                matecat_async
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mbotTestedDecoder/scripts/generic/multi-bleu.perl"
               data-name="mbotTestedDecoder"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mbotTestedDecoder
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mbotdecoder/scripts/generic/multi-bleu.perl"
               data-name="mbotdecoder"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mbotdecoder
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/merge-cmd/scripts/generic/multi-bleu.perl"
               data-name="merge-cmd"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                merge-cmd
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mert-mtm5/scripts/generic/multi-bleu.perl"
               data-name="mert-mtm5"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mert-mtm5
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mert-other_metrics/scripts/generic/multi-bleu.perl"
               data-name="mert-other_metrics"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mert-other_metrics
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mert-otherMetrics/scripts/generic/multi-bleu.perl"
               data-name="mert-otherMetrics"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mert-otherMetrics
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mingw-release-2.1.1/scripts/generic/multi-bleu.perl"
               data-name="mingw-release-2.1.1"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mingw-release-2.1.1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/minphr/scripts/generic/multi-bleu.perl"
               data-name="minphr"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                minphr
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mira-mtm5/scripts/generic/multi-bleu.perl"
               data-name="mira-mtm5"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mira-mtm5
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/miramerge/scripts/generic/multi-bleu.perl"
               data-name="miramerge"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                miramerge
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mjdenkowski/scripts/generic/multi-bleu.perl"
               data-name="mjdenkowski"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mjdenkowski
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mmt-dev/scripts/generic/multi-bleu.perl"
               data-name="mmt-dev"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mmt-dev
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mmt-merge/scripts/generic/multi-bleu.perl"
               data-name="mmt-merge"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mmt-merge
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/moses-batchedlm/scripts/generic/multi-bleu.perl"
               data-name="moses-batchedlm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                moses-batchedlm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/moses-ebay.premerge/scripts/generic/multi-bleu.perl"
               data-name="moses-ebay.premerge"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                moses-ebay.premerge
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/moses-ebay/scripts/generic/multi-bleu.perl"
               data-name="moses-ebay"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                moses-ebay
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/moses-server/scripts/generic/multi-bleu.perl"
               data-name="moses-server"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                moses-server
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/moses+batchedlm/scripts/generic/multi-bleu.perl"
               data-name="moses+batchedlm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                moses+batchedlm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/moses_cachebased/scripts/generic/multi-bleu.perl"
               data-name="moses_cachebased"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                moses_cachebased
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mt3_chart/scripts/generic/multi-bleu.perl"
               data-name="mt3_chart"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mt3_chart
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mt3_constraint/scripts/generic/multi-bleu.perl"
               data-name="mt3_constraint"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mt3_constraint
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mtm_lm/scripts/generic/multi-bleu.perl"
               data-name="mtm_lm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mtm_lm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mtm6-parallel-mert/scripts/generic/multi-bleu.perl"
               data-name="mtm6-parallel-mert"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mtm6-parallel-mert
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mtm6_cruise_control/scripts/generic/multi-bleu.perl"
               data-name="mtm6_cruise_control"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mtm6_cruise_control
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/mtm15-multipass/scripts/generic/multi-bleu.perl"
               data-name="mtm15-multipass"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                mtm15-multipass
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/multiple-models/scripts/generic/multi-bleu.perl"
               data-name="multiple-models"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                multiple-models
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/multiple-models%403284/scripts/generic/multi-bleu.perl"
               data-name="multiple-models@3284"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                multiple-models@3284
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/neuralpt/scripts/generic/multi-bleu.perl"
               data-name="neuralpt"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                neuralpt
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/ni_DPR_reordering_model/scripts/generic/multi-bleu.perl"
               data-name="ni_DPR_reordering_model"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ni_DPR_reordering_model
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/ni_DPR_reordering_model2/scripts/generic/multi-bleu.perl"
               data-name="ni_DPR_reordering_model2"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ni_DPR_reordering_model2
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/nmt-hybrid/scripts/generic/multi-bleu.perl"
               data-name="nmt-hybrid"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nmt-hybrid
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/norestart/scripts/generic/multi-bleu.perl"
               data-name="norestart"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                norestart
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/nplm/scripts/generic/multi-bleu.perl"
               data-name="nplm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                nplm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/oovpt/scripts/generic/multi-bleu.perl"
               data-name="oovpt"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                oovpt
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/patch-detokenizer-ko/scripts/generic/multi-bleu.perl"
               data-name="patch-detokenizer-ko"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                patch-detokenizer-ko
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/perf_baseline0/scripts/generic/multi-bleu.perl"
               data-name="perf_baseline0"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                perf_baseline0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/perf_moses2.baseline/scripts/generic/multi-bleu.perl"
               data-name="perf_moses2.baseline"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                perf_moses2.baseline
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/perf_moses2.gpulm/scripts/generic/multi-bleu.perl"
               data-name="perf_moses2.gpulm"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                perf_moses2.gpulm
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/prefix-private-cderscorer-member/scripts/generic/multi-bleu.perl"
               data-name="prefix-private-cderscorer-member"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                prefix-private-cderscorer-member
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/probing/scripts/generic/multi-bleu.perl"
               data-name="probing"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                probing
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/revert-176-RELEASE-3.0-cpp14/scripts/generic/multi-bleu.perl"
               data-name="revert-176-RELEASE-3.0-cpp14"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                revert-176-RELEASE-3.0-cpp14
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/samplerank/scripts/generic/multi-bleu.perl"
               data-name="samplerank"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                samplerank
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/span-length/scripts/generic/multi-bleu.perl"
               data-name="span-length"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                span-length
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/sparse-reordering/scripts/generic/multi-bleu.perl"
               data-name="sparse-reordering"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                sparse-reordering
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/suffix_array/scripts/generic/multi-bleu.perl"
               data-name="suffix_array"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                suffix_array
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/syntaxContext/scripts/generic/multi-bleu.perl"
               data-name="syntaxContext"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                syntaxContext
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/ug-debug/scripts/generic/multi-bleu.perl"
               data-name="ug-debug"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ug-debug
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/ug-m4m/scripts/generic/multi-bleu.perl"
               data-name="ug-m4m"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ug-m4m
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/ug-mct-find/scripts/generic/multi-bleu.perl"
               data-name="ug-mct-find"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                ug-mct-find
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/vw_integration/scripts/generic/multi-bleu.perl"
               data-name="vw_integration"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                vw_integration
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/vw_tgtcontext/scripts/generic/multi-bleu.perl"
               data-name="vw_tgtcontext"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                vw_tgtcontext
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/wipo/scripts/generic/multi-bleu.perl"
               data-name="wipo"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                wipo
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/wipoNew/scripts/generic/multi-bleu.perl"
               data-name="wipoNew"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                wipoNew
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/wipoNewMerger/scripts/generic/multi-bleu.perl"
               data-name="wipoNewMerger"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                wipoNewMerger
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/yanggao-softdep-hiero/scripts/generic/multi-bleu.perl"
               data-name="yanggao-softdep-hiero"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                yanggao-softdep-hiero
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
               href="/moses-smt/mosesdecoder/blob/yanggao_maxent/scripts/generic/multi-bleu.perl"
               data-name="yanggao_maxent"
               data-skip-pjax="true"
               rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target js-select-menu-filter-text">
                yanggao_maxent
              </span>
            </a>
        </div>

          <div class="select-menu-no-results">Nothing to show</div>
      </div>

      <div class="select-menu-list select-menu-tab-bucket js-select-menu-tab-bucket" data-tab-filter="tags">
        <div data-filterable-for="context-commitish-filter-field" data-filterable-type="substring">


            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/ranked-sampling-v0.1.0/scripts/generic/multi-bleu.perl"
              data-name="ranked-sampling-v0.1.0"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="ranked-sampling-v0.1.0">
                ranked-sampling-v0.1.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/pre-MMT/scripts/generic/multi-bleu.perl"
              data-name="pre-MMT"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="pre-MMT">
                pre-MMT
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/mtm_before_cube_pruning/scripts/generic/multi-bleu.perl"
              data-name="mtm_before_cube_pruning"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="mtm_before_cube_pruning">
                mtm_before_cube_pruning
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/mmt-mvp-v0.12.1/scripts/generic/multi-bleu.perl"
              data-name="mmt-mvp-v0.12.1"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="mmt-mvp-v0.12.1">
                mmt-mvp-v0.12.1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/mmt-mvp-v0.12.0/scripts/generic/multi-bleu.perl"
              data-name="mmt-mvp-v0.12.0"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="mmt-mvp-v0.12.0">
                mmt-mvp-v0.12.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/mmt-mvp-v0.9.0/scripts/generic/multi-bleu.perl"
              data-name="mmt-mvp-v0.9.0"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="mmt-mvp-v0.9.0">
                mmt-mvp-v0.9.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/mmt-mvp-v0.2.1/scripts/generic/multi-bleu.perl"
              data-name="mmt-mvp-v0.2.1"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="mmt-mvp-v0.2.1">
                mmt-mvp-v0.2.1
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/mmt-mvp-0.2.0/scripts/generic/multi-bleu.perl"
              data-name="mmt-mvp-0.2.0"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="mmt-mvp-0.2.0">
                mmt-mvp-0.2.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/mira-mtm5_on-the-fly_paramater_updates_working/scripts/generic/multi-bleu.perl"
              data-name="mira-mtm5_on-the-fly_paramater_updates_working"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="mira-mtm5_on-the-fly_paramater_updates_working">
                mira-mtm5_on-the-fly_paramater_updates_working
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/eva_2d_pruning%401521/scripts/generic/multi-bleu.perl"
              data-name="eva_2d_pruning@1521"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="eva_2d_pruning@1521">
                eva_2d_pruning@1521
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/eva_2d_pruning%401488/scripts/generic/multi-bleu.perl"
              data-name="eva_2d_pruning@1488"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="eva_2d_pruning@1488">
                eva_2d_pruning@1488
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/eva_2d_pruning/scripts/generic/multi-bleu.perl"
              data-name="eva_2d_pruning"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="eva_2d_pruning">
                eva_2d_pruning
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/START_JHU/scripts/generic/multi-bleu.perl"
              data-name="START_JHU"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="START_JHU">
                START_JHU
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/SCRIPTS-RELEASE-20060802-1746/scripts/generic/multi-bleu.perl"
              data-name="SCRIPTS-RELEASE-20060802-1746"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="SCRIPTS-RELEASE-20060802-1746">
                SCRIPTS-RELEASE-20060802-1746
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/SCRIPTS-RELEASE-20060801-1856/scripts/generic/multi-bleu.perl"
              data-name="SCRIPTS-RELEASE-20060801-1856"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="SCRIPTS-RELEASE-20060801-1856">
                SCRIPTS-RELEASE-20060801-1856
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/SCRIPTS-RELEASE-20060731-2118/scripts/generic/multi-bleu.perl"
              data-name="SCRIPTS-RELEASE-20060731-2118"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="SCRIPTS-RELEASE-20060731-2118">
                SCRIPTS-RELEASE-20060731-2118
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/SCRIPTS-RELEASE-20060731-1701/scripts/generic/multi-bleu.perl"
              data-name="SCRIPTS-RELEASE-20060731-1701"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="SCRIPTS-RELEASE-20060731-1701">
                SCRIPTS-RELEASE-20060731-1701
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/Root_HIEUS_BRANCH_2006_08_15/scripts/generic/multi-bleu.perl"
              data-name="Root_HIEUS_BRANCH_2006_08_15"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="Root_HIEUS_BRANCH_2006_08_15">
                Root_HIEUS_BRANCH_2006_08_15
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/RELEASE-2006-07-26/scripts/generic/multi-bleu.perl"
              data-name="RELEASE-2006-07-26"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="RELEASE-2006-07-26">
                RELEASE-2006-07-26
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/RELEASE-2006-07-14/scripts/generic/multi-bleu.perl"
              data-name="RELEASE-2006-07-14"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="RELEASE-2006-07-14">
                RELEASE-2006-07-14
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/RELEASE-4.0/scripts/generic/multi-bleu.perl"
              data-name="RELEASE-4.0"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="RELEASE-4.0">
                RELEASE-4.0
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/LAST-SINGLE-MODEL-ONLY-2010-08-09/scripts/generic/multi-bleu.perl"
              data-name="LAST-SINGLE-MODEL-ONLY-2010-08-09"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="LAST-SINGLE-MODEL-ONLY-2010-08-09">
                LAST-SINGLE-MODEL-ONLY-2010-08-09
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/LAST-PHRASE-BASED-ONLY-2010-04-08/scripts/generic/multi-bleu.perl"
              data-name="LAST-PHRASE-BASED-ONLY-2010-04-08"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="LAST-PHRASE-BASED-ONLY-2010-04-08">
                LAST-PHRASE-BASED-ONLY-2010-04-08
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/HIEU_THESIS/scripts/generic/multi-bleu.perl"
              data-name="HIEU_THESIS"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="HIEU_THESIS">
                HIEU_THESIS
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/BEFORE_MERGE_MAIN/scripts/generic/multi-bleu.perl"
              data-name="BEFORE_MERGE_MAIN"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="BEFORE_MERGE_MAIN">
                BEFORE_MERGE_MAIN
              </span>
            </a>
            <a class="select-menu-item js-navigation-item js-navigation-open "
              href="/moses-smt/mosesdecoder/tree/ABHISHEK_THESIS/scripts/generic/multi-bleu.perl"
              data-name="ABHISHEK_THESIS"
              data-skip-pjax="true"
              rel="nofollow">
              <svg class="octicon octicon-check select-menu-item-icon" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M12 5l-8 8-4-4 1.5-1.5L4 10l6.5-6.5L12 5z"/></svg>
              <span class="select-menu-item-text css-truncate-target" title="ABHISHEK_THESIS">
                ABHISHEK_THESIS
              </span>
            </a>
        </div>

        <div class="select-menu-no-results">Nothing to show</div>
      </div>

    </div>
  </div>
</div>

    <div class="BtnGroup float-right">
      <a href="/moses-smt/mosesdecoder/find/master"
            class="js-pjax-capture-input btn btn-sm BtnGroup-item"
            data-pjax
            data-hotkey="t">
        Find file
      </a>
      <clipboard-copy for="blob-path" class="btn btn-sm BtnGroup-item">
        Copy path
      </clipboard-copy>
    </div>
    <div id="blob-path" class="breadcrumb">
      <span class="repo-root js-repo-root"><span class="js-path-segment"><a data-pjax="true" href="/moses-smt/mosesdecoder"><span>mosesdecoder</span></a></span></span><span class="separator">/</span><span class="js-path-segment"><a data-pjax="true" href="/moses-smt/mosesdecoder/tree/master/scripts"><span>scripts</span></a></span><span class="separator">/</span><span class="js-path-segment"><a data-pjax="true" href="/moses-smt/mosesdecoder/tree/master/scripts/generic"><span>generic</span></a></span><span class="separator">/</span><strong class="final-path">multi-bleu.perl</strong>
    </div>
  </div>


  <include-fragment src="/moses-smt/mosesdecoder/contributors/master/scripts/generic/multi-bleu.perl" class="commit-tease">
    <div>
      Fetching contributors&hellip;
    </div>

    <div class="commit-tease-contributors">
      <img alt="" class="loader-loading float-left" src="https://assets-cdn.github.com/images/spinners/octocat-spinner-32-EAF2F5.gif" width="16" height="16" />
      <span class="loader-error">Cannot retrieve contributors at this time</span>
    </div>
</include-fragment>


  <div class="file">
    <div class="file-header">
  <div class="file-actions">

    <div class="BtnGroup">
      <a id="raw-url" class="btn btn-sm BtnGroup-item" href="/moses-smt/mosesdecoder/raw/master/scripts/generic/multi-bleu.perl">Raw</a>
        <a class="btn btn-sm js-update-url-with-hash BtnGroup-item" data-hotkey="b" href="/moses-smt/mosesdecoder/blame/master/scripts/generic/multi-bleu.perl">Blame</a>
      <a rel="nofollow" class="btn btn-sm BtnGroup-item" href="/moses-smt/mosesdecoder/commits/master/scripts/generic/multi-bleu.perl">History</a>
    </div>


          <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form js-update-url-with-hash" action="/moses-smt/mosesdecoder/edit/master/scripts/generic/multi-bleu.perl" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="Fw0VZGlHrpMxZVcjVZEjp6GPt2+TZeJ825grqWyjGy0sYIHnHUipFQ36a/WHgrwl3SC3CJ+Gj8SCqDpyb1US9w==" />
            <button class="btn-octicon tooltipped tooltipped-nw" type="submit"
              aria-label="Fork this project and edit the file" data-hotkey="e" data-disable-with>
              <svg class="octicon octicon-pencil" viewBox="0 0 14 16" version="1.1" width="14" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M0 12v3h3l8-8-3-3-8 8zm3 2H1v-2h1v1h1v1zm10.3-9.3L12 6 9 3l1.3-1.3a.996.996 0 0 1 1.41 0l1.59 1.59c.39.39.39 1.02 0 1.41z"/></svg>
            </button>
</form>
        <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="inline-form" action="/moses-smt/mosesdecoder/delete/master/scripts/generic/multi-bleu.perl" accept-charset="UTF-8" method="post"><input name="utf8" type="hidden" value="&#x2713;" /><input type="hidden" name="authenticity_token" value="IhR0Zf+QnM6TmY+MgFHhwgdR2Y4Lx+ciWefP5INZwkdZaDKcU1z3dEZhXxYSAILJ0Gv8I5yWdf7y4tyIbNc82w==" />
          <button class="btn-octicon btn-octicon-danger tooltipped tooltipped-nw" type="submit"
            aria-label="Fork this project and delete the file" data-disable-with>
            <svg class="octicon octicon-trashcan" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M11 2H9c0-.55-.45-1-1-1H5c-.55 0-1 .45-1 1H2c-.55 0-1 .45-1 1v1c0 .55.45 1 1 1v9c0 .55.45 1 1 1h7c.55 0 1-.45 1-1V5c.55 0 1-.45 1-1V3c0-.55-.45-1-1-1zm-1 12H3V5h1v8h1V5h1v8h1V5h1v8h1V5h1v9zm1-10H2V3h9v1z"/></svg>
          </button>
</form>  </div>

  <div class="file-info">
      <span class="file-mode" title="File mode">executable file</span>
      <span class="file-info-divider"></span>
      178 lines (160 sloc)
      <span class="file-info-divider"></span>
    5.11 KB
  </div>
</div>

    

  <div itemprop="text" class="blob-wrapper data type-perl">
      <table class="highlight tab-size js-file-line-container" data-tab-size="8">
      <tr>
        <td id="L1" class="blob-num js-line-number" data-line-number="1"></td>
        <td id="LC1" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>!/usr/bin/env perl</span></td>
      </tr>
      <tr>
        <td id="L2" class="blob-num js-line-number" data-line-number="2"></td>
        <td id="LC2" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span></span></td>
      </tr>
      <tr>
        <td id="L3" class="blob-num js-line-number" data-line-number="3"></td>
        <td id="LC3" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> This file is part of moses.  Its use is licensed under the GNU Lesser General</span></td>
      </tr>
      <tr>
        <td id="L4" class="blob-num js-line-number" data-line-number="4"></td>
        <td id="LC4" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> Public License version 2.1 or, at your option, any later version.</span></td>
      </tr>
      <tr>
        <td id="L5" class="blob-num js-line-number" data-line-number="5"></td>
        <td id="LC5" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L6" class="blob-num js-line-number" data-line-number="6"></td>
        <td id="LC6" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> $Id$</span></td>
      </tr>
      <tr>
        <td id="L7" class="blob-num js-line-number" data-line-number="7"></td>
        <td id="LC7" class="blob-code blob-code-inner js-file-line"><span class="pl-k">use</span> warnings;</td>
      </tr>
      <tr>
        <td id="L8" class="blob-num js-line-number" data-line-number="8"></td>
        <td id="LC8" class="blob-code blob-code-inner js-file-line"><span class="pl-k">use</span> strict;</td>
      </tr>
      <tr>
        <td id="L9" class="blob-num js-line-number" data-line-number="9"></td>
        <td id="LC9" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L10" class="blob-num js-line-number" data-line-number="10"></td>
        <td id="LC10" class="blob-code blob-code-inner js-file-line"><span class="pl-k">my</span> <span class="pl-smi">$lowercase</span> = 0;</td>
      </tr>
      <tr>
        <td id="L11" class="blob-num js-line-number" data-line-number="11"></td>
        <td id="LC11" class="blob-code blob-code-inner js-file-line"><span class="pl-k">if</span> (<span class="pl-smi">$ARGV</span>[0] <span class="pl-c1">eq</span> <span class="pl-s"><span class="pl-pds">&quot;</span>-lc<span class="pl-pds">&quot;</span></span>) {</td>
      </tr>
      <tr>
        <td id="L12" class="blob-num js-line-number" data-line-number="12"></td>
        <td id="LC12" class="blob-code blob-code-inner js-file-line">  <span class="pl-smi">$lowercase</span> = 1;</td>
      </tr>
      <tr>
        <td id="L13" class="blob-num js-line-number" data-line-number="13"></td>
        <td id="LC13" class="blob-code blob-code-inner js-file-line">  <span class="pl-c1">shift</span>;</td>
      </tr>
      <tr>
        <td id="L14" class="blob-num js-line-number" data-line-number="14"></td>
        <td id="LC14" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L15" class="blob-num js-line-number" data-line-number="15"></td>
        <td id="LC15" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L16" class="blob-num js-line-number" data-line-number="16"></td>
        <td id="LC16" class="blob-code blob-code-inner js-file-line"><span class="pl-k">my</span> <span class="pl-smi">$stem</span> = <span class="pl-smi">$ARGV</span>[0];</td>
      </tr>
      <tr>
        <td id="L17" class="blob-num js-line-number" data-line-number="17"></td>
        <td id="LC17" class="blob-code blob-code-inner js-file-line"><span class="pl-k">if</span> (!<span class="pl-c1">defined</span> <span class="pl-smi">$stem</span>) {</td>
      </tr>
      <tr>
        <td id="L18" class="blob-num js-line-number" data-line-number="18"></td>
        <td id="LC18" class="blob-code blob-code-inner js-file-line">  <span class="pl-c1">print</span> <span class="pl-c1">STDERR</span> <span class="pl-s"><span class="pl-pds">&quot;</span>usage: multi-bleu.pl [-lc] reference &lt; hypothesis<span class="pl-cce">\n</span><span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L19" class="blob-num js-line-number" data-line-number="19"></td>
        <td id="LC19" class="blob-code blob-code-inner js-file-line">  <span class="pl-c1">print</span> <span class="pl-c1">STDERR</span> <span class="pl-s"><span class="pl-pds">&quot;</span>Reads the references from reference or reference0, reference1, ...<span class="pl-cce">\n</span><span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L20" class="blob-num js-line-number" data-line-number="20"></td>
        <td id="LC20" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">exit</span>(1);</td>
      </tr>
      <tr>
        <td id="L21" class="blob-num js-line-number" data-line-number="21"></td>
        <td id="LC21" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L22" class="blob-num js-line-number" data-line-number="22"></td>
        <td id="LC22" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L23" class="blob-num js-line-number" data-line-number="23"></td>
        <td id="LC23" class="blob-code blob-code-inner js-file-line"><span class="pl-smi">$stem</span> .= <span class="pl-s"><span class="pl-pds">&quot;</span>.ref<span class="pl-pds">&quot;</span></span> <span class="pl-k">if</span> !<span class="pl-k">-e</span> <span class="pl-smi">$stem</span> &amp;&amp; !<span class="pl-k">-e</span> <span class="pl-smi">$stem</span>.<span class="pl-s"><span class="pl-pds">&quot;</span>0<span class="pl-pds">&quot;</span></span> &amp;&amp; <span class="pl-k">-e</span> <span class="pl-smi">$stem</span>.<span class="pl-s"><span class="pl-pds">&quot;</span>.ref0<span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L24" class="blob-num js-line-number" data-line-number="24"></td>
        <td id="LC24" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L25" class="blob-num js-line-number" data-line-number="25"></td>
        <td id="LC25" class="blob-code blob-code-inner js-file-line"><span class="pl-k">my</span> <span class="pl-smi">@REF</span>;</td>
      </tr>
      <tr>
        <td id="L26" class="blob-num js-line-number" data-line-number="26"></td>
        <td id="LC26" class="blob-code blob-code-inner js-file-line"><span class="pl-k">my</span> <span class="pl-smi">$ref</span>=0;</td>
      </tr>
      <tr>
        <td id="L27" class="blob-num js-line-number" data-line-number="27"></td>
        <td id="LC27" class="blob-code blob-code-inner js-file-line"><span class="pl-k">while</span>(<span class="pl-k">-e</span> <span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-smi">$stem$ref</span><span class="pl-pds">&quot;</span></span>) {</td>
      </tr>
      <tr>
        <td id="L28" class="blob-num js-line-number" data-line-number="28"></td>
        <td id="LC28" class="blob-code blob-code-inner js-file-line">    &amp;add_to_ref(<span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-smi">$stem$ref</span><span class="pl-pds">&quot;</span></span>,\<span class="pl-smi">@REF</span>);</td>
      </tr>
      <tr>
        <td id="L29" class="blob-num js-line-number" data-line-number="29"></td>
        <td id="LC29" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$ref</span>++;</td>
      </tr>
      <tr>
        <td id="L30" class="blob-num js-line-number" data-line-number="30"></td>
        <td id="LC30" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L31" class="blob-num js-line-number" data-line-number="31"></td>
        <td id="LC31" class="blob-code blob-code-inner js-file-line">&amp;add_to_ref(<span class="pl-smi">$stem</span>,\<span class="pl-smi">@REF</span>) <span class="pl-k">if</span> <span class="pl-k">-e</span> <span class="pl-smi">$stem</span>;</td>
      </tr>
      <tr>
        <td id="L32" class="blob-num js-line-number" data-line-number="32"></td>
        <td id="LC32" class="blob-code blob-code-inner js-file-line"><span class="pl-k">die</span>(<span class="pl-s"><span class="pl-pds">&quot;</span>ERROR: could not find reference file <span class="pl-smi">$stem</span><span class="pl-pds">&quot;</span></span>) <span class="pl-k">unless</span> <span class="pl-c1">scalar</span> <span class="pl-smi">@REF</span>;</td>
      </tr>
      <tr>
        <td id="L33" class="blob-num js-line-number" data-line-number="33"></td>
        <td id="LC33" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L34" class="blob-num js-line-number" data-line-number="34"></td>
        <td id="LC34" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span> add additional references explicitly specified on the command line</span></td>
      </tr>
      <tr>
        <td id="L35" class="blob-num js-line-number" data-line-number="35"></td>
        <td id="LC35" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">shift</span>;</td>
      </tr>
      <tr>
        <td id="L36" class="blob-num js-line-number" data-line-number="36"></td>
        <td id="LC36" class="blob-code blob-code-inner js-file-line"><span class="pl-k">foreach</span> <span class="pl-k">my</span> <span class="pl-smi">$stem</span> (<span class="pl-smi">@ARGV</span>) {</td>
      </tr>
      <tr>
        <td id="L37" class="blob-num js-line-number" data-line-number="37"></td>
        <td id="LC37" class="blob-code blob-code-inner js-file-line">    &amp;add_to_ref(<span class="pl-smi">$stem</span>,\<span class="pl-smi">@REF</span>) <span class="pl-k">if</span> <span class="pl-k">-e</span> <span class="pl-smi">$stem</span>;</td>
      </tr>
      <tr>
        <td id="L38" class="blob-num js-line-number" data-line-number="38"></td>
        <td id="LC38" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L39" class="blob-num js-line-number" data-line-number="39"></td>
        <td id="LC39" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L40" class="blob-num js-line-number" data-line-number="40"></td>
        <td id="LC40" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L41" class="blob-num js-line-number" data-line-number="41"></td>
        <td id="LC41" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L42" class="blob-num js-line-number" data-line-number="42"></td>
        <td id="LC42" class="blob-code blob-code-inner js-file-line"><span class="pl-k">sub</span> <span class="pl-en">add_to_ref</span> {</td>
      </tr>
      <tr>
        <td id="L43" class="blob-num js-line-number" data-line-number="43"></td>
        <td id="LC43" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">my</span> (<span class="pl-smi">$file</span>,<span class="pl-smi">$REF</span>) = <span class="pl-smi">@_</span>;</td>
      </tr>
      <tr>
        <td id="L44" class="blob-num js-line-number" data-line-number="44"></td>
        <td id="LC44" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">my</span> <span class="pl-smi">$s</span>=0;</td>
      </tr>
      <tr>
        <td id="L45" class="blob-num js-line-number" data-line-number="45"></td>
        <td id="LC45" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">if</span> (<span class="pl-smi">$file</span> =~ <span class="pl-sr"><span class="pl-pds">/</span>.gz<span class="pl-k">$</span><span class="pl-pds">/</span></span>) {</td>
      </tr>
      <tr>
        <td id="L46" class="blob-num js-line-number" data-line-number="46"></td>
        <td id="LC46" class="blob-code blob-code-inner js-file-line">	<span class="pl-c1">open</span>(REF,<span class="pl-s"><span class="pl-pds">&quot;</span>gzip -dc <span class="pl-smi">$file</span>|<span class="pl-pds">&quot;</span></span>) <span class="pl-k">or</span> <span class="pl-k">die</span> <span class="pl-s"><span class="pl-pds">&quot;</span>Can&#39;t read <span class="pl-smi">$file</span><span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L47" class="blob-num js-line-number" data-line-number="47"></td>
        <td id="LC47" class="blob-code blob-code-inner js-file-line">    } <span class="pl-k">else</span> { </td>
      </tr>
      <tr>
        <td id="L48" class="blob-num js-line-number" data-line-number="48"></td>
        <td id="LC48" class="blob-code blob-code-inner js-file-line">	<span class="pl-c1">open</span>(REF,<span class="pl-smi">$file</span>) <span class="pl-k">or</span> <span class="pl-k">die</span> <span class="pl-s"><span class="pl-pds">&quot;</span>Can&#39;t read <span class="pl-smi">$file</span><span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L49" class="blob-num js-line-number" data-line-number="49"></td>
        <td id="LC49" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L50" class="blob-num js-line-number" data-line-number="50"></td>
        <td id="LC50" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">while</span>(&lt;REF&gt;) {</td>
      </tr>
      <tr>
        <td id="L51" class="blob-num js-line-number" data-line-number="51"></td>
        <td id="LC51" class="blob-code blob-code-inner js-file-line">	<span class="pl-c1">chop</span>;</td>
      </tr>
      <tr>
        <td id="L52" class="blob-num js-line-number" data-line-number="52"></td>
        <td id="LC52" class="blob-code blob-code-inner js-file-line">	<span class="pl-c1">push</span> @{<span class="pl-smi">$$REF</span>[<span class="pl-smi">$s</span>++]}, <span class="pl-smi">$_</span>;</td>
      </tr>
      <tr>
        <td id="L53" class="blob-num js-line-number" data-line-number="53"></td>
        <td id="LC53" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L54" class="blob-num js-line-number" data-line-number="54"></td>
        <td id="LC54" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">close</span>(REF);</td>
      </tr>
      <tr>
        <td id="L55" class="blob-num js-line-number" data-line-number="55"></td>
        <td id="LC55" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L56" class="blob-num js-line-number" data-line-number="56"></td>
        <td id="LC56" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L57" class="blob-num js-line-number" data-line-number="57"></td>
        <td id="LC57" class="blob-code blob-code-inner js-file-line"><span class="pl-k">my</span>(<span class="pl-smi">@CORRECT</span>,<span class="pl-smi">@TOTAL</span>,<span class="pl-smi">$length_translation</span>,<span class="pl-smi">$length_reference</span>);</td>
      </tr>
      <tr>
        <td id="L58" class="blob-num js-line-number" data-line-number="58"></td>
        <td id="LC58" class="blob-code blob-code-inner js-file-line"><span class="pl-k">my</span> <span class="pl-smi">$s</span>=0;</td>
      </tr>
      <tr>
        <td id="L59" class="blob-num js-line-number" data-line-number="59"></td>
        <td id="LC59" class="blob-code blob-code-inner js-file-line"><span class="pl-k">while</span>(&lt;<span class="pl-c1">STDIN</span>&gt;) {</td>
      </tr>
      <tr>
        <td id="L60" class="blob-num js-line-number" data-line-number="60"></td>
        <td id="LC60" class="blob-code blob-code-inner js-file-line">    <span class="pl-c1">chop</span>;</td>
      </tr>
      <tr>
        <td id="L61" class="blob-num js-line-number" data-line-number="61"></td>
        <td id="LC61" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$_</span> = <span class="pl-c1">lc</span> <span class="pl-k">if</span> <span class="pl-smi">$lowercase</span>;</td>
      </tr>
      <tr>
        <td id="L62" class="blob-num js-line-number" data-line-number="62"></td>
        <td id="LC62" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">my</span> <span class="pl-smi">@WORD</span> = <span class="pl-c1">split</span>;</td>
      </tr>
      <tr>
        <td id="L63" class="blob-num js-line-number" data-line-number="63"></td>
        <td id="LC63" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">my</span> <span class="pl-smi">%REF_NGRAM</span> = ();</td>
      </tr>
      <tr>
        <td id="L64" class="blob-num js-line-number" data-line-number="64"></td>
        <td id="LC64" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">my</span> <span class="pl-smi">$length_translation_this_sentence</span> = <span class="pl-c1">scalar</span>(<span class="pl-smi">@WORD</span>);</td>
      </tr>
      <tr>
        <td id="L65" class="blob-num js-line-number" data-line-number="65"></td>
        <td id="LC65" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">my</span> (<span class="pl-smi">$closest_diff</span>,<span class="pl-smi">$closest_length</span>) = (9999,9999);</td>
      </tr>
      <tr>
        <td id="L66" class="blob-num js-line-number" data-line-number="66"></td>
        <td id="LC66" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">foreach</span> <span class="pl-k">my</span> <span class="pl-smi">$reference</span> (@{<span class="pl-smi">$REF</span>[<span class="pl-smi">$s</span>]}) {</td>
      </tr>
      <tr>
        <td id="L67" class="blob-num js-line-number" data-line-number="67"></td>
        <td id="LC67" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>      print &quot;$s $_ &lt;=&gt; $reference\n&quot;;</span></td>
      </tr>
      <tr>
        <td id="L68" class="blob-num js-line-number" data-line-number="68"></td>
        <td id="LC68" class="blob-code blob-code-inner js-file-line">  <span class="pl-smi">$reference</span> = <span class="pl-c1">lc</span>(<span class="pl-smi">$reference</span>) <span class="pl-k">if</span> <span class="pl-smi">$lowercase</span>;</td>
      </tr>
      <tr>
        <td id="L69" class="blob-num js-line-number" data-line-number="69"></td>
        <td id="LC69" class="blob-code blob-code-inner js-file-line">	<span class="pl-k">my</span> <span class="pl-smi">@WORD</span> = <span class="pl-c1">split</span>(<span class="pl-s"><span class="pl-pds">&#39;</span> <span class="pl-pds">&#39;</span></span>,<span class="pl-smi">$reference</span>);</td>
      </tr>
      <tr>
        <td id="L70" class="blob-num js-line-number" data-line-number="70"></td>
        <td id="LC70" class="blob-code blob-code-inner js-file-line">	<span class="pl-k">my</span> <span class="pl-smi">$length</span> = <span class="pl-c1">scalar</span>(<span class="pl-smi">@WORD</span>);</td>
      </tr>
      <tr>
        <td id="L71" class="blob-num js-line-number" data-line-number="71"></td>
        <td id="LC71" class="blob-code blob-code-inner js-file-line">        <span class="pl-k">my</span> <span class="pl-smi">$diff</span> = abs(<span class="pl-smi">$length_translation_this_sentence</span>-<span class="pl-smi">$length</span>);</td>
      </tr>
      <tr>
        <td id="L72" class="blob-num js-line-number" data-line-number="72"></td>
        <td id="LC72" class="blob-code blob-code-inner js-file-line">	<span class="pl-k">if</span> (<span class="pl-smi">$diff</span> &lt; <span class="pl-smi">$closest_diff</span>) {</td>
      </tr>
      <tr>
        <td id="L73" class="blob-num js-line-number" data-line-number="73"></td>
        <td id="LC73" class="blob-code blob-code-inner js-file-line">	    <span class="pl-smi">$closest_diff</span> = <span class="pl-smi">$diff</span>;</td>
      </tr>
      <tr>
        <td id="L74" class="blob-num js-line-number" data-line-number="74"></td>
        <td id="LC74" class="blob-code blob-code-inner js-file-line">	    <span class="pl-smi">$closest_length</span> = <span class="pl-smi">$length</span>;</td>
      </tr>
      <tr>
        <td id="L75" class="blob-num js-line-number" data-line-number="75"></td>
        <td id="LC75" class="blob-code blob-code-inner js-file-line">	    <span class="pl-c"><span class="pl-c">#</span> print STDERR &quot;$s: closest diff &quot;.abs($length_translation_this_sentence-$length).&quot; = abs($length_translation_this_sentence-$length), setting len: $closest_length\n&quot;;</span></td>
      </tr>
      <tr>
        <td id="L76" class="blob-num js-line-number" data-line-number="76"></td>
        <td id="LC76" class="blob-code blob-code-inner js-file-line">	} <span class="pl-k">elsif</span> (<span class="pl-smi">$diff</span> == <span class="pl-smi">$closest_diff</span>) {</td>
      </tr>
      <tr>
        <td id="L77" class="blob-num js-line-number" data-line-number="77"></td>
        <td id="LC77" class="blob-code blob-code-inner js-file-line">            <span class="pl-smi">$closest_length</span> = <span class="pl-smi">$length</span> <span class="pl-k">if</span> <span class="pl-smi">$length</span> &lt; <span class="pl-smi">$closest_length</span>;</td>
      </tr>
      <tr>
        <td id="L78" class="blob-num js-line-number" data-line-number="78"></td>
        <td id="LC78" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> from two references with the same closeness to me</span></td>
      </tr>
      <tr>
        <td id="L79" class="blob-num js-line-number" data-line-number="79"></td>
        <td id="LC79" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> take the *shorter* into account, not the &quot;first&quot; one.</span></td>
      </tr>
      <tr>
        <td id="L80" class="blob-num js-line-number" data-line-number="80"></td>
        <td id="LC80" class="blob-code blob-code-inner js-file-line">        }</td>
      </tr>
      <tr>
        <td id="L81" class="blob-num js-line-number" data-line-number="81"></td>
        <td id="LC81" class="blob-code blob-code-inner js-file-line">	<span class="pl-k">for</span>(<span class="pl-k">my</span> <span class="pl-smi">$n</span>=1;<span class="pl-smi">$n</span>&lt;=4;<span class="pl-smi">$n</span>++) {</td>
      </tr>
      <tr>
        <td id="L82" class="blob-num js-line-number" data-line-number="82"></td>
        <td id="LC82" class="blob-code blob-code-inner js-file-line">	    <span class="pl-k">my</span> <span class="pl-smi">%REF_NGRAM_N</span> = ();</td>
      </tr>
      <tr>
        <td id="L83" class="blob-num js-line-number" data-line-number="83"></td>
        <td id="LC83" class="blob-code blob-code-inner js-file-line">	    <span class="pl-k">for</span>(<span class="pl-k">my</span> <span class="pl-smi">$start</span>=0;<span class="pl-smi">$start</span>&lt;=<span class="pl-smi">$#WORD</span>-(<span class="pl-smi">$n</span>-1);<span class="pl-smi">$start</span>++) {</td>
      </tr>
      <tr>
        <td id="L84" class="blob-num js-line-number" data-line-number="84"></td>
        <td id="LC84" class="blob-code blob-code-inner js-file-line">		<span class="pl-k">my</span> <span class="pl-smi">$ngram</span> = <span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-smi">$n</span><span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L85" class="blob-num js-line-number" data-line-number="85"></td>
        <td id="LC85" class="blob-code blob-code-inner js-file-line">		<span class="pl-k">for</span>(<span class="pl-k">my</span> <span class="pl-smi">$w</span>=0;<span class="pl-smi">$w</span>&lt;<span class="pl-smi">$n</span>;<span class="pl-smi">$w</span>++) {</td>
      </tr>
      <tr>
        <td id="L86" class="blob-num js-line-number" data-line-number="86"></td>
        <td id="LC86" class="blob-code blob-code-inner js-file-line">		    <span class="pl-smi">$ngram</span> .= <span class="pl-s"><span class="pl-pds">&quot;</span> <span class="pl-pds">&quot;</span></span>.<span class="pl-smi">$WORD</span>[<span class="pl-smi">$start</span>+<span class="pl-smi">$w</span>];</td>
      </tr>
      <tr>
        <td id="L87" class="blob-num js-line-number" data-line-number="87"></td>
        <td id="LC87" class="blob-code blob-code-inner js-file-line">		}</td>
      </tr>
      <tr>
        <td id="L88" class="blob-num js-line-number" data-line-number="88"></td>
        <td id="LC88" class="blob-code blob-code-inner js-file-line">		<span class="pl-smi">$REF_NGRAM_N</span>{<span class="pl-smi">$ngram</span>}++;</td>
      </tr>
      <tr>
        <td id="L89" class="blob-num js-line-number" data-line-number="89"></td>
        <td id="LC89" class="blob-code blob-code-inner js-file-line">	    }</td>
      </tr>
      <tr>
        <td id="L90" class="blob-num js-line-number" data-line-number="90"></td>
        <td id="LC90" class="blob-code blob-code-inner js-file-line">	    <span class="pl-k">foreach</span> <span class="pl-k">my</span> <span class="pl-smi">$ngram</span> (<span class="pl-c1">keys</span> <span class="pl-smi">%REF_NGRAM_N</span>) {</td>
      </tr>
      <tr>
        <td id="L91" class="blob-num js-line-number" data-line-number="91"></td>
        <td id="LC91" class="blob-code blob-code-inner js-file-line">		<span class="pl-k">if</span> (!<span class="pl-c1">defined</span>(<span class="pl-smi">$REF_NGRAM</span>{<span class="pl-smi">$ngram</span>}) ||</td>
      </tr>
      <tr>
        <td id="L92" class="blob-num js-line-number" data-line-number="92"></td>
        <td id="LC92" class="blob-code blob-code-inner js-file-line">		    <span class="pl-smi">$REF_NGRAM</span>{<span class="pl-smi">$ngram</span>} &lt; <span class="pl-smi">$REF_NGRAM_N</span>{<span class="pl-smi">$ngram</span>}) {</td>
      </tr>
      <tr>
        <td id="L93" class="blob-num js-line-number" data-line-number="93"></td>
        <td id="LC93" class="blob-code blob-code-inner js-file-line">		    <span class="pl-smi">$REF_NGRAM</span>{<span class="pl-smi">$ngram</span>} = <span class="pl-smi">$REF_NGRAM_N</span>{<span class="pl-smi">$ngram</span>};</td>
      </tr>
      <tr>
        <td id="L94" class="blob-num js-line-number" data-line-number="94"></td>
        <td id="LC94" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>	    print &quot;$i: REF_NGRAM{$ngram} = $REF_NGRAM{$ngram}&lt;BR&gt;\n&quot;;</span></td>
      </tr>
      <tr>
        <td id="L95" class="blob-num js-line-number" data-line-number="95"></td>
        <td id="LC95" class="blob-code blob-code-inner js-file-line">		}</td>
      </tr>
      <tr>
        <td id="L96" class="blob-num js-line-number" data-line-number="96"></td>
        <td id="LC96" class="blob-code blob-code-inner js-file-line">	    }</td>
      </tr>
      <tr>
        <td id="L97" class="blob-num js-line-number" data-line-number="97"></td>
        <td id="LC97" class="blob-code blob-code-inner js-file-line">	}</td>
      </tr>
      <tr>
        <td id="L98" class="blob-num js-line-number" data-line-number="98"></td>
        <td id="LC98" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L99" class="blob-num js-line-number" data-line-number="99"></td>
        <td id="LC99" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$length_translation</span> += <span class="pl-smi">$length_translation_this_sentence</span>;</td>
      </tr>
      <tr>
        <td id="L100" class="blob-num js-line-number" data-line-number="100"></td>
        <td id="LC100" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$length_reference</span> += <span class="pl-smi">$closest_length</span>;</td>
      </tr>
      <tr>
        <td id="L101" class="blob-num js-line-number" data-line-number="101"></td>
        <td id="LC101" class="blob-code blob-code-inner js-file-line">    <span class="pl-k">for</span>(<span class="pl-k">my</span> <span class="pl-smi">$n</span>=1;<span class="pl-smi">$n</span>&lt;=4;<span class="pl-smi">$n</span>++) {</td>
      </tr>
      <tr>
        <td id="L102" class="blob-num js-line-number" data-line-number="102"></td>
        <td id="LC102" class="blob-code blob-code-inner js-file-line">	<span class="pl-k">my</span> <span class="pl-smi">%T_NGRAM</span> = ();</td>
      </tr>
      <tr>
        <td id="L103" class="blob-num js-line-number" data-line-number="103"></td>
        <td id="LC103" class="blob-code blob-code-inner js-file-line">	<span class="pl-k">for</span>(<span class="pl-k">my</span> <span class="pl-smi">$start</span>=0;<span class="pl-smi">$start</span>&lt;=<span class="pl-smi">$#WORD</span>-(<span class="pl-smi">$n</span>-1);<span class="pl-smi">$start</span>++) {</td>
      </tr>
      <tr>
        <td id="L104" class="blob-num js-line-number" data-line-number="104"></td>
        <td id="LC104" class="blob-code blob-code-inner js-file-line">	    <span class="pl-k">my</span> <span class="pl-smi">$ngram</span> = <span class="pl-s"><span class="pl-pds">&quot;</span><span class="pl-smi">$n</span><span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L105" class="blob-num js-line-number" data-line-number="105"></td>
        <td id="LC105" class="blob-code blob-code-inner js-file-line">	    <span class="pl-k">for</span>(<span class="pl-k">my</span> <span class="pl-smi">$w</span>=0;<span class="pl-smi">$w</span>&lt;<span class="pl-smi">$n</span>;<span class="pl-smi">$w</span>++) {</td>
      </tr>
      <tr>
        <td id="L106" class="blob-num js-line-number" data-line-number="106"></td>
        <td id="LC106" class="blob-code blob-code-inner js-file-line">		<span class="pl-smi">$ngram</span> .= <span class="pl-s"><span class="pl-pds">&quot;</span> <span class="pl-pds">&quot;</span></span>.<span class="pl-smi">$WORD</span>[<span class="pl-smi">$start</span>+<span class="pl-smi">$w</span>];</td>
      </tr>
      <tr>
        <td id="L107" class="blob-num js-line-number" data-line-number="107"></td>
        <td id="LC107" class="blob-code blob-code-inner js-file-line">	    }</td>
      </tr>
      <tr>
        <td id="L108" class="blob-num js-line-number" data-line-number="108"></td>
        <td id="LC108" class="blob-code blob-code-inner js-file-line">	    <span class="pl-smi">$T_NGRAM</span>{<span class="pl-smi">$ngram</span>}++;</td>
      </tr>
      <tr>
        <td id="L109" class="blob-num js-line-number" data-line-number="109"></td>
        <td id="LC109" class="blob-code blob-code-inner js-file-line">	}</td>
      </tr>
      <tr>
        <td id="L110" class="blob-num js-line-number" data-line-number="110"></td>
        <td id="LC110" class="blob-code blob-code-inner js-file-line">	<span class="pl-k">foreach</span> <span class="pl-k">my</span> <span class="pl-smi">$ngram</span> (<span class="pl-c1">keys</span> <span class="pl-smi">%T_NGRAM</span>) {</td>
      </tr>
      <tr>
        <td id="L111" class="blob-num js-line-number" data-line-number="111"></td>
        <td id="LC111" class="blob-code blob-code-inner js-file-line">	    <span class="pl-smi">$ngram</span> =~ <span class="pl-sr"><span class="pl-pds">/</span>^(<span class="pl-cce">\d</span>+) <span class="pl-pds">/</span></span>;</td>
      </tr>
      <tr>
        <td id="L112" class="blob-num js-line-number" data-line-number="112"></td>
        <td id="LC112" class="blob-code blob-code-inner js-file-line">	    <span class="pl-k">my</span> <span class="pl-smi">$n</span> = <span class="pl-smi">$1</span>;</td>
      </tr>
      <tr>
        <td id="L113" class="blob-num js-line-number" data-line-number="113"></td>
        <td id="LC113" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> my $corr = 0;</span></td>
      </tr>
      <tr>
        <td id="L114" class="blob-num js-line-number" data-line-number="114"></td>
        <td id="LC114" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>	print &quot;$i e $ngram $T_NGRAM{$ngram}&lt;BR&gt;\n&quot;;</span></td>
      </tr>
      <tr>
        <td id="L115" class="blob-num js-line-number" data-line-number="115"></td>
        <td id="LC115" class="blob-code blob-code-inner js-file-line">	    <span class="pl-smi">$TOTAL</span>[<span class="pl-smi">$n</span>] += <span class="pl-smi">$T_NGRAM</span>{<span class="pl-smi">$ngram</span>};</td>
      </tr>
      <tr>
        <td id="L116" class="blob-num js-line-number" data-line-number="116"></td>
        <td id="LC116" class="blob-code blob-code-inner js-file-line">	    <span class="pl-k">if</span> (<span class="pl-c1">defined</span>(<span class="pl-smi">$REF_NGRAM</span>{<span class="pl-smi">$ngram</span>})) {</td>
      </tr>
      <tr>
        <td id="L117" class="blob-num js-line-number" data-line-number="117"></td>
        <td id="LC117" class="blob-code blob-code-inner js-file-line">		<span class="pl-k">if</span> (<span class="pl-smi">$REF_NGRAM</span>{<span class="pl-smi">$ngram</span>} &gt;= <span class="pl-smi">$T_NGRAM</span>{<span class="pl-smi">$ngram</span>}) {</td>
      </tr>
      <tr>
        <td id="L118" class="blob-num js-line-number" data-line-number="118"></td>
        <td id="LC118" class="blob-code blob-code-inner js-file-line">		    <span class="pl-smi">$CORRECT</span>[<span class="pl-smi">$n</span>] += <span class="pl-smi">$T_NGRAM</span>{<span class="pl-smi">$ngram</span>};</td>
      </tr>
      <tr>
        <td id="L119" class="blob-num js-line-number" data-line-number="119"></td>
        <td id="LC119" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c"><span class="pl-c">#</span> $corr =  $T_NGRAM{$ngram};</span></td>
      </tr>
      <tr>
        <td id="L120" class="blob-num js-line-number" data-line-number="120"></td>
        <td id="LC120" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>	    print &quot;$i e correct1 $T_NGRAM{$ngram}&lt;BR&gt;\n&quot;;</span></td>
      </tr>
      <tr>
        <td id="L121" class="blob-num js-line-number" data-line-number="121"></td>
        <td id="LC121" class="blob-code blob-code-inner js-file-line">		}</td>
      </tr>
      <tr>
        <td id="L122" class="blob-num js-line-number" data-line-number="122"></td>
        <td id="LC122" class="blob-code blob-code-inner js-file-line">		<span class="pl-k">else</span> {</td>
      </tr>
      <tr>
        <td id="L123" class="blob-num js-line-number" data-line-number="123"></td>
        <td id="LC123" class="blob-code blob-code-inner js-file-line">		    <span class="pl-smi">$CORRECT</span>[<span class="pl-smi">$n</span>] += <span class="pl-smi">$REF_NGRAM</span>{<span class="pl-smi">$ngram</span>};</td>
      </tr>
      <tr>
        <td id="L124" class="blob-num js-line-number" data-line-number="124"></td>
        <td id="LC124" class="blob-code blob-code-inner js-file-line">                    <span class="pl-c"><span class="pl-c">#</span> $corr =  $REF_NGRAM{$ngram};</span></td>
      </tr>
      <tr>
        <td id="L125" class="blob-num js-line-number" data-line-number="125"></td>
        <td id="LC125" class="blob-code blob-code-inner js-file-line"><span class="pl-c"><span class="pl-c">#</span>	    print &quot;$i e correct2 $REF_NGRAM{$ngram}&lt;BR&gt;\n&quot;;</span></td>
      </tr>
      <tr>
        <td id="L126" class="blob-num js-line-number" data-line-number="126"></td>
        <td id="LC126" class="blob-code blob-code-inner js-file-line">		}</td>
      </tr>
      <tr>
        <td id="L127" class="blob-num js-line-number" data-line-number="127"></td>
        <td id="LC127" class="blob-code blob-code-inner js-file-line">	    }</td>
      </tr>
      <tr>
        <td id="L128" class="blob-num js-line-number" data-line-number="128"></td>
        <td id="LC128" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> $REF_NGRAM{$ngram} = 0 if !defined $REF_NGRAM{$ngram};</span></td>
      </tr>
      <tr>
        <td id="L129" class="blob-num js-line-number" data-line-number="129"></td>
        <td id="LC129" class="blob-code blob-code-inner js-file-line">            <span class="pl-c"><span class="pl-c">#</span> print STDERR &quot;$ngram: {$s, $REF_NGRAM{$ngram}, $T_NGRAM{$ngram}, $corr}\n&quot;</span></td>
      </tr>
      <tr>
        <td id="L130" class="blob-num js-line-number" data-line-number="130"></td>
        <td id="LC130" class="blob-code blob-code-inner js-file-line">	}</td>
      </tr>
      <tr>
        <td id="L131" class="blob-num js-line-number" data-line-number="131"></td>
        <td id="LC131" class="blob-code blob-code-inner js-file-line">    }</td>
      </tr>
      <tr>
        <td id="L132" class="blob-num js-line-number" data-line-number="132"></td>
        <td id="LC132" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$s</span>++;</td>
      </tr>
      <tr>
        <td id="L133" class="blob-num js-line-number" data-line-number="133"></td>
        <td id="LC133" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L134" class="blob-num js-line-number" data-line-number="134"></td>
        <td id="LC134" class="blob-code blob-code-inner js-file-line"><span class="pl-k">my</span> <span class="pl-smi">$brevity_penalty</span> = 1;</td>
      </tr>
      <tr>
        <td id="L135" class="blob-num js-line-number" data-line-number="135"></td>
        <td id="LC135" class="blob-code blob-code-inner js-file-line"><span class="pl-k">my</span> <span class="pl-smi">$bleu</span> = 0;</td>
      </tr>
      <tr>
        <td id="L136" class="blob-num js-line-number" data-line-number="136"></td>
        <td id="LC136" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L137" class="blob-num js-line-number" data-line-number="137"></td>
        <td id="LC137" class="blob-code blob-code-inner js-file-line"><span class="pl-k">my</span> <span class="pl-smi">@bleu</span>=();</td>
      </tr>
      <tr>
        <td id="L138" class="blob-num js-line-number" data-line-number="138"></td>
        <td id="LC138" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L139" class="blob-num js-line-number" data-line-number="139"></td>
        <td id="LC139" class="blob-code blob-code-inner js-file-line"><span class="pl-k">for</span>(<span class="pl-k">my</span> <span class="pl-smi">$n</span>=1;<span class="pl-smi">$n</span>&lt;=4;<span class="pl-smi">$n</span>++) {</td>
      </tr>
      <tr>
        <td id="L140" class="blob-num js-line-number" data-line-number="140"></td>
        <td id="LC140" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">if</span> (<span class="pl-c1">defined</span> (<span class="pl-smi">$TOTAL</span>[<span class="pl-smi">$n</span>])){</td>
      </tr>
      <tr>
        <td id="L141" class="blob-num js-line-number" data-line-number="141"></td>
        <td id="LC141" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$bleu</span>[<span class="pl-smi">$n</span>]=(<span class="pl-smi">$TOTAL</span>[<span class="pl-smi">$n</span>])?<span class="pl-smi">$CORRECT</span>[<span class="pl-smi">$n</span>]/<span class="pl-smi">$TOTAL</span>[<span class="pl-smi">$n</span>]:0;</td>
      </tr>
      <tr>
        <td id="L142" class="blob-num js-line-number" data-line-number="142"></td>
        <td id="LC142" class="blob-code blob-code-inner js-file-line">    <span class="pl-c"><span class="pl-c">#</span> print STDERR &quot;CORRECT[$n]:$CORRECT[$n] TOTAL[$n]:$TOTAL[$n]\n&quot;;</span></td>
      </tr>
      <tr>
        <td id="L143" class="blob-num js-line-number" data-line-number="143"></td>
        <td id="LC143" class="blob-code blob-code-inner js-file-line">  }<span class="pl-k">else</span>{</td>
      </tr>
      <tr>
        <td id="L144" class="blob-num js-line-number" data-line-number="144"></td>
        <td id="LC144" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$bleu</span>[<span class="pl-smi">$n</span>]=0;</td>
      </tr>
      <tr>
        <td id="L145" class="blob-num js-line-number" data-line-number="145"></td>
        <td id="LC145" class="blob-code blob-code-inner js-file-line">  }</td>
      </tr>
      <tr>
        <td id="L146" class="blob-num js-line-number" data-line-number="146"></td>
        <td id="LC146" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L147" class="blob-num js-line-number" data-line-number="147"></td>
        <td id="LC147" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L148" class="blob-num js-line-number" data-line-number="148"></td>
        <td id="LC148" class="blob-code blob-code-inner js-file-line"><span class="pl-k">if</span> (<span class="pl-smi">$length_reference</span>==0){</td>
      </tr>
      <tr>
        <td id="L149" class="blob-num js-line-number" data-line-number="149"></td>
        <td id="LC149" class="blob-code blob-code-inner js-file-line">  <span class="pl-c1">printf</span> <span class="pl-s"><span class="pl-pds">&quot;</span>BLEU = 0, 0/0/0/0 (BP=0, ratio=0, hyp_len=0, ref_len=0)<span class="pl-cce">\n</span><span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L150" class="blob-num js-line-number" data-line-number="150"></td>
        <td id="LC150" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">exit</span>(1);</td>
      </tr>
      <tr>
        <td id="L151" class="blob-num js-line-number" data-line-number="151"></td>
        <td id="LC151" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L152" class="blob-num js-line-number" data-line-number="152"></td>
        <td id="LC152" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L153" class="blob-num js-line-number" data-line-number="153"></td>
        <td id="LC153" class="blob-code blob-code-inner js-file-line"><span class="pl-k">if</span> (<span class="pl-smi">$length_translation</span>&lt;<span class="pl-smi">$length_reference</span>) {</td>
      </tr>
      <tr>
        <td id="L154" class="blob-num js-line-number" data-line-number="154"></td>
        <td id="LC154" class="blob-code blob-code-inner js-file-line">  <span class="pl-smi">$brevity_penalty</span> = <span class="pl-c1">exp</span>(1-<span class="pl-smi">$length_reference</span>/<span class="pl-smi">$length_translation</span>);</td>
      </tr>
      <tr>
        <td id="L155" class="blob-num js-line-number" data-line-number="155"></td>
        <td id="LC155" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
      <tr>
        <td id="L156" class="blob-num js-line-number" data-line-number="156"></td>
        <td id="LC156" class="blob-code blob-code-inner js-file-line"><span class="pl-smi">$bleu</span> = <span class="pl-smi">$brevity_penalty</span> * <span class="pl-c1">exp</span>((my_log( <span class="pl-smi">$bleu</span>[1] ) +</td>
      </tr>
      <tr>
        <td id="L157" class="blob-num js-line-number" data-line-number="157"></td>
        <td id="LC157" class="blob-code blob-code-inner js-file-line">				my_log( <span class="pl-smi">$bleu</span>[2] ) +</td>
      </tr>
      <tr>
        <td id="L158" class="blob-num js-line-number" data-line-number="158"></td>
        <td id="LC158" class="blob-code blob-code-inner js-file-line">				my_log( <span class="pl-smi">$bleu</span>[3] ) +</td>
      </tr>
      <tr>
        <td id="L159" class="blob-num js-line-number" data-line-number="159"></td>
        <td id="LC159" class="blob-code blob-code-inner js-file-line">				my_log( <span class="pl-smi">$bleu</span>[4] ) ) / 4) ;</td>
      </tr>
      <tr>
        <td id="L160" class="blob-num js-line-number" data-line-number="160"></td>
        <td id="LC160" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">printf</span> <span class="pl-s"><span class="pl-pds">&quot;</span>BLEU = %.2f, %.1f/%.1f/%.1f/%.1f (BP=%.3f, ratio=%.3f, hyp_len=<span class="pl-smi">%d</span>, ref_len=<span class="pl-smi">%d</span>)<span class="pl-cce">\n</span><span class="pl-pds">&quot;</span></span>,</td>
      </tr>
      <tr>
        <td id="L161" class="blob-num js-line-number" data-line-number="161"></td>
        <td id="LC161" class="blob-code blob-code-inner js-file-line">    100*<span class="pl-smi">$bleu</span>,</td>
      </tr>
      <tr>
        <td id="L162" class="blob-num js-line-number" data-line-number="162"></td>
        <td id="LC162" class="blob-code blob-code-inner js-file-line">    100*<span class="pl-smi">$bleu</span>[1],</td>
      </tr>
      <tr>
        <td id="L163" class="blob-num js-line-number" data-line-number="163"></td>
        <td id="LC163" class="blob-code blob-code-inner js-file-line">    100*<span class="pl-smi">$bleu</span>[2],</td>
      </tr>
      <tr>
        <td id="L164" class="blob-num js-line-number" data-line-number="164"></td>
        <td id="LC164" class="blob-code blob-code-inner js-file-line">    100*<span class="pl-smi">$bleu</span>[3],</td>
      </tr>
      <tr>
        <td id="L165" class="blob-num js-line-number" data-line-number="165"></td>
        <td id="LC165" class="blob-code blob-code-inner js-file-line">    100*<span class="pl-smi">$bleu</span>[4],</td>
      </tr>
      <tr>
        <td id="L166" class="blob-num js-line-number" data-line-number="166"></td>
        <td id="LC166" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$brevity_penalty</span>,</td>
      </tr>
      <tr>
        <td id="L167" class="blob-num js-line-number" data-line-number="167"></td>
        <td id="LC167" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$length_translation</span> / <span class="pl-smi">$length_reference</span>,</td>
      </tr>
      <tr>
        <td id="L168" class="blob-num js-line-number" data-line-number="168"></td>
        <td id="LC168" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$length_translation</span>,</td>
      </tr>
      <tr>
        <td id="L169" class="blob-num js-line-number" data-line-number="169"></td>
        <td id="LC169" class="blob-code blob-code-inner js-file-line">    <span class="pl-smi">$length_reference</span>;</td>
      </tr>
      <tr>
        <td id="L170" class="blob-num js-line-number" data-line-number="170"></td>
        <td id="LC170" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L171" class="blob-num js-line-number" data-line-number="171"></td>
        <td id="LC171" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L172" class="blob-num js-line-number" data-line-number="172"></td>
        <td id="LC172" class="blob-code blob-code-inner js-file-line"><span class="pl-c1">print</span> <span class="pl-c1">STDERR</span> <span class="pl-s"><span class="pl-pds">&quot;</span>It is in-advisable to publish scores from multi-bleu.perl.  The scores depend on your tokenizer, which is unlikely to be reproducible from your paper or consistent across research groups.  Instead you should detokenize then use mteval-v14.pl, which has a standard tokenization.  Scores from multi-bleu.perl can still be used for internal purposes when you have a consistent tokenizer.<span class="pl-cce">\n</span><span class="pl-pds">&quot;</span></span>;</td>
      </tr>
      <tr>
        <td id="L173" class="blob-num js-line-number" data-line-number="173"></td>
        <td id="LC173" class="blob-code blob-code-inner js-file-line">
</td>
      </tr>
      <tr>
        <td id="L174" class="blob-num js-line-number" data-line-number="174"></td>
        <td id="LC174" class="blob-code blob-code-inner js-file-line"><span class="pl-k">sub</span> <span class="pl-en">my_log</span> {</td>
      </tr>
      <tr>
        <td id="L175" class="blob-num js-line-number" data-line-number="175"></td>
        <td id="LC175" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">return</span> -9999999999 <span class="pl-k">unless</span> <span class="pl-smi">$_</span>[0];</td>
      </tr>
      <tr>
        <td id="L176" class="blob-num js-line-number" data-line-number="176"></td>
        <td id="LC176" class="blob-code blob-code-inner js-file-line">  <span class="pl-k">return</span> <span class="pl-c1">log</span>(<span class="pl-smi">$_</span>[0]);</td>
      </tr>
      <tr>
        <td id="L177" class="blob-num js-line-number" data-line-number="177"></td>
        <td id="LC177" class="blob-code blob-code-inner js-file-line">}</td>
      </tr>
</table>

  <div class="BlobToolbar position-absolute js-file-line-actions dropdown js-menu-container js-select-menu d-none" aria-hidden="true">
    <button class="btn-octicon ml-0 px-2 p-0 bg-white border border-gray-dark rounded-1 dropdown-toggle js-menu-target" type="button" aria-expanded="false" aria-haspopup="true" aria-label="Inline file action toolbar" aria-controls="inline-file-actions">
      <svg class="octicon octicon-kebab-horizontal" viewBox="0 0 13 16" version="1.1" width="13" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M1.5 9a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zm5 0a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3zM13 7.5a1.5 1.5 0 1 1-3 0 1.5 1.5 0 0 1 3 0z"/></svg>
    </button>
    <div class="dropdown-menu-content js-menu-content" id="inline-file-actions">
      <ul class="BlobToolbar-dropdown dropdown-menu dropdown-menu-se mt-2">
        <li><clipboard-copy class="dropdown-item" id="js-copy-lines" style="cursor:pointer;" data-original-text="Copy lines">Copy lines</clipboard-copy></li>
        <li><clipboard-copy class="dropdown-item" id="js-copy-permalink" style="cursor:pointer;" data-original-text="Copy permalink">Copy permalink</clipboard-copy></li>
        <li><a class="dropdown-item js-update-url-with-hash" id="js-view-git-blame" href="/moses-smt/mosesdecoder/blame/b56d97dee0e9e452d2ffa08e809a90917ad7e7a7/scripts/generic/multi-bleu.perl">View git blame</a></li>
          <li><a class="dropdown-item" id="js-new-issue" href="/moses-smt/mosesdecoder/issues/new">Open new issue</a></li>
      </ul>
    </div>
  </div>

  </div>

  </div>

  <button type="button" data-facebox="#jump-to-line" data-facebox-class="linejump" data-hotkey="l" class="d-none">Jump to Line</button>
  <div id="jump-to-line" style="display:none">
    <!-- '"` --><!-- </textarea></xmp> --></option></form><form class="js-jump-to-line-form" action="" accept-charset="UTF-8" method="get"><input name="utf8" type="hidden" value="&#x2713;" />
      <input class="form-control linejump-input js-jump-to-line-field" type="text" placeholder="Jump to line&hellip;" aria-label="Jump to line" autofocus>
      <button type="submit" class="btn">Go</button>
</form>  </div>


  </div>
  <div class="modal-backdrop js-touch-events"></div>
</div>

    </div>
  </div>

  </div>

      
<div class="footer container-lg px-3" role="contentinfo">
  <div class="position-relative d-flex flex-justify-between pt-6 pb-2 mt-6 f6 text-gray border-top border-gray-light ">
    <ul class="list-style-none d-flex flex-wrap ">
      <li class="mr-3">&copy; 2018 <span title="0.39556s from unicorn-2360624063-7tqhs">GitHub</span>, Inc.</li>
        <li class="mr-3"><a data-ga-click="Footer, go to terms, text:terms" href="https://github.com/site/terms">Terms</a></li>
        <li class="mr-3"><a data-ga-click="Footer, go to privacy, text:privacy" href="https://github.com/site/privacy">Privacy</a></li>
        <li class="mr-3"><a href="https://help.github.com/articles/github-security/" data-ga-click="Footer, go to security, text:security">Security</a></li>
        <li class="mr-3"><a href="https://status.github.com/" data-ga-click="Footer, go to status, text:status">Status</a></li>
        <li><a data-ga-click="Footer, go to help, text:help" href="https://help.github.com">Help</a></li>
    </ul>

    <a aria-label="Homepage" title="GitHub" class="footer-octicon" href="https://github.com">
      <svg height="24" class="octicon octicon-mark-github" viewBox="0 0 16 16" version="1.1" width="24" aria-hidden="true"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
</a>
   <ul class="list-style-none d-flex flex-wrap ">
        <li class="mr-3"><a data-ga-click="Footer, go to contact, text:contact" href="https://github.com/contact">Contact GitHub</a></li>
      <li class="mr-3"><a href="https://developer.github.com" data-ga-click="Footer, go to api, text:api">API</a></li>
      <li class="mr-3"><a href="https://training.github.com" data-ga-click="Footer, go to training, text:training">Training</a></li>
      <li class="mr-3"><a href="https://shop.github.com" data-ga-click="Footer, go to shop, text:shop">Shop</a></li>
        <li class="mr-3"><a href="https://blog.github.com" data-ga-click="Footer, go to blog, text:blog">Blog</a></li>
        <li><a data-ga-click="Footer, go to about, text:about" href="https://github.com/about">About</a></li>

    </ul>
  </div>
  <div class="d-flex flex-justify-center pb-6">
    <span class="f6 text-gray-light"></span>
  </div>
</div>



  <div id="ajax-error-message" class="ajax-error-message flash flash-error">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <button type="button" class="flash-close js-ajax-error-dismiss" aria-label="Dismiss error">
      <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
    </button>
    You can’t perform that action at this time.
  </div>


    <script crossorigin="anonymous" integrity="sha512-2GVr5rsbbfKbHM6oRrri41+qJ2ltJBCqluASS29fj+9yHGLFmFhq0C64VMdL57UJ34G2+FXU+8FZhaAOnsCEhw==" type="application/javascript" src="https://assets-cdn.github.com/assets/compat-bb7abfb15ed4ffb0da9056d4c980fba5.js"></script>
    <script crossorigin="anonymous" integrity="sha512-HD3VGNUZdKcyhzxthfJME6V4ByRTPdCGqlqq7sPm8jFweQQjz00nAJzWMAR9GUDiP2yYWqSxM6JxNxc2J/4rGQ==" type="application/javascript" src="https://assets-cdn.github.com/assets/frameworks-2afbd5df19ed1f9aaf0a04c15e9d0b35.js"></script>
    
    <script crossorigin="anonymous" async="async" integrity="sha512-IdHQYUq4tkFFl+J/TM87/2TTheiMcB5bJfeRbGRfac3jQMr9DHMNsiuFC/1Vm+tTh3OwkssVx5Tk0iZIpfbZuQ==" type="application/javascript" src="https://assets-cdn.github.com/assets/github-d67ce02b55f439df4644fda3f6efab0a.js"></script>
    
    
    
  <div class="js-stale-session-flash stale-session-flash flash flash-warn flash-banner d-none">
    <svg class="octicon octicon-alert" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M8.893 1.5c-.183-.31-.52-.5-.887-.5s-.703.19-.886.5L.138 13.499a.98.98 0 0 0 0 1.001c.193.31.53.501.886.501h13.964c.367 0 .704-.19.877-.5a1.03 1.03 0 0 0 .01-1.002L8.893 1.5zm.133 11.497H6.987v-2.003h2.039v2.003zm0-3.004H6.987V5.987h2.039v4.006z"/></svg>
    <span class="signed-in-tab-flash">You signed in with another tab or window. <a href="">Reload</a> to refresh your session.</span>
    <span class="signed-out-tab-flash">You signed out in another tab or window. <a href="">Reload</a> to refresh your session.</span>
  </div>
  <div class="facebox" id="facebox" style="display:none;">
  <div class="facebox-popup">
    <div class="facebox-content" role="dialog" aria-labelledby="facebox-header" aria-describedby="facebox-description">
    </div>
    <button type="button" class="facebox-close js-facebox-close" aria-label="Close modal">
      <svg class="octicon octicon-x" viewBox="0 0 12 16" version="1.1" width="12" height="16" aria-hidden="true"><path fill-rule="evenodd" d="M7.48 8l3.75 3.75-1.48 1.48L6 9.48l-3.75 3.75-1.48-1.48L4.52 8 .77 4.25l1.48-1.48L6 6.52l3.75-3.75 1.48 1.48L7.48 8z"/></svg>
    </button>
  </div>
</div>

  <div class="Popover js-hovercard-content position-absolute" style="display: none; outline: none;" tabindex="0">
  <div class="Popover-message Popover-message--bottom-left Popover-message--large Box box-shadow-large" style="width:360px;">
  </div>
</div>

<div id="hovercard-aria-description" class="sr-only">
  Press h to open a hovercard with more details.
</div>


  </body>
</html>

