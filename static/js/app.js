/**
 * LevelFind — PC Game Recommendation System JS Application Code
 */

// 1. STATE TERPUSAT (SINGLE SOURCE OF TRUTH)
let currentLang = localStorage.getItem('lf-lang') || 'id';

function setLang(lang, animate = false) {
    currentLang = lang;
    localStorage.setItem('lf-lang', lang);
    document.documentElement.setAttribute('lang', lang === 'en' ? 'en' : 'id');

    const $langBtn = $('#lang-btn');
    const $langLabel = $('#lang-label');

    if (animate) {
        $langBtn.addClass('rotating');
        $langLabel.addClass('flip');
        setTimeout(() => {
            $langLabel.text(lang.toUpperCase());
            $langLabel.removeClass('flip');
            $langBtn.removeClass('rotating');
        }, 200);
    } else {
        $langLabel.text(lang.toUpperCase());
    }

    const $i18nElements = $('[data-i18n]');
    const $placeholders = $('[data-i18n-placeholder]');
    const $xaiCards = $('.rec-card');

    if (animate) {
        $i18nElements.addClass('i18n-transition i18n-fade-out');
        $placeholders.addClass('placeholder-transition placeholder-fade');
        $xaiCards.find('.xai-text').addClass('i18n-transition i18n-fade-out');

        setTimeout(() => {
            updateDOMText(lang);
            $i18nElements.removeClass('i18n-fade-out');
            $placeholders.removeClass('placeholder-fade');
            $xaiCards.find('.xai-text').removeClass('i18n-fade-out');
        }, 200);
    } else {
        $i18nElements.addClass('i18n-transition');
        updateDOMText(lang);
    }
}

// MATCH THRESHOLDS & HELPER
const MATCH_THRESHOLDS = { HIGH: 80, MEDIUM: 60 };

function getMatchTier(simPct) {
    let score = parseFloat(simPct) || 0;
    if (score >= MATCH_THRESHOLDS.HIGH) {
        return { tier: 'high', color: 'var(--gold-bright)', badgeClass: 'badge-match-high' };
    } else if (score >= MATCH_THRESHOLDS.MEDIUM) {
        return { tier: 'med', color: 'var(--gold)', badgeClass: 'badge-match-med' };
    } else {
        return { tier: 'low', color: 'var(--text-3)', badgeClass: 'badge-match-low' };
    }
}

// BILINGUAL DICTIONARY
const i18nDict = {
    id: {
        "hero-title": "Mulai dengan satu game<br>yang kamu <em>suka</em>",
        "hero-sub": "Sistem AI kami menganalisis ribuan game untuk menemukan rekomendasi paling akurat berdasarkan game favorit Anda.",
        "search-placeholder": "Contoh: Elden Ring, Palworld...",
        "search-btn": "Temukan",
        "history-label": "Terakhir Dicari:",
        "suggestion-title": "Coba salah satu game populer berikut:",
        "stat-games": "Game",
        "stat-algo": "Algoritma",
        "stat-sim": "Similarity",
        "analysis-result": "Hasil Analisis",
        "main-game": "Main Game",
        "game-profile": "Profil & Karakteristik Game",
        "profile-sub": "Sistem mempelajari dan mencocokkan game ini dengan game lain berdasarkan informasi berikut:",
        "synopsis-label": "Sinopsis / Deskripsi Game",
        "fallback_desc": "Informasi detail untuk game ini belum tersedia di sistem kami.",
        "synopsis-fallback": "Sinopsis lengkap untuk game ini belum tersedia di sistem kami. Game ini dikategorikan berdasarkan fitur &amp; genre:",
        "desc-fallback": "Informasi detail untuk game ini belum tersedia di sistem kami.",
        "categories-label": "Kategori & Fitur (Tags & Genres)",
        "open-steam": "Buka di Steam",
        "similar-rec": "Rekomendasi Serupa",
        "share-btn": "Share",
        "genre-all": "Semua Genre",
        "price-all": "Semua Harga",
        "price-free": "Gratis (Free to Play)",
        "price-under10": "Di bawah $10",
        "price-under30": "Di bawah $30",
        "price-above30": "Di atas $30",
        "sort-match": "Highest Match",
        "sort-rating": "Best Rating",
        "sort-price": "Lowest Price",
        "no-search-title": "Belum ada pencarian",
        "no-search-sub": "Mulai dengan satu game yang kamu suka — sisanya biar kami yang urus.",
        "not-found-title": "Game Tidak Ditemukan",
        "not-found-sub": "Coba periksa kembali ejaan Anda atau masukkan judul game populer lain.",
        "rank-1": "Game Referensi Anda",
        "read-more": "Baca selengkapnya",
        "show-less": "Sembunyikan",
        "indexed": "Indexed",
        "detail-btn": "Lihat Detail &#8594;",
        "loading-text": "Mencari game yang cocok...",
        "modal-title": "Cara Kerja Sistem LevelFind",
        "modal-intro": "Sistem ini merekomendasikan game berdasarkan kemiripan konten melalui 3 langkah sederhana:",
        "modal-step1": "<strong>Pembersihan Data:</strong> Sistem merapikan teks deskripsi, tag, dan genre game agar mudah dibaca oleh mesin.",
        "modal-step2": "<strong>Pemberian Nilai Kata:</strong> Sistem mencari kata kunci unik dari sebuah game yang paling membedakannya dari game lain.",
        "modal-step3": "<strong>Pencocokan Kemiripan:</strong> Sistem menghitung seberapa mirip kata kunci game target dengan ribuan game lain di sistem kami untuk menemukan kecocokan terbaik.",
        "modal-note": "*Persentase Kecocokan menunjukkan seberapa identik elemen cerita, mekanik, dan atmosfer game dibandingkan dengan judul pencarian Anda.",
        "rating-na": "<i class=\"fas fa-minus-circle\" style=\"color: var(--text-3);\"></i> N/A (Belum ada ulasan yang cukup)",
        "xai-title": "Penjelasan Rekomendasi (XAI)",
        "xai_title": "Penjelasan Rekomendasi (XAI)",
        "gmodal-features": "Kategori & Fitur Irisan:",
        "modal-desc-fallback": "Informasi detail untuk game ini belum tersedia di sistem kami.",
        "load-more-btn": "Tampilkan Lebih Banyak",
        "similarity_label": "Tingkat Kemiripan",
        "similarity_level": "Tingkat Kemiripan",
        "top_contrib_title": "Kecocokan Utama",
        "other_shared_title": "Kesamaan Lainnya",
        "calc_method_title": "Sistem Cerdas",
        "calc_method_desc": "TF-IDF + Cosine Similarity",
        "view_xai_btn": "Lihat Penjelasan XAI",
        "tight_cluster_badge": "Skor Serupa",
        "synopsis-lang-notice-text": "Sinopsis ini ditampilkan dalam Bahasa Inggris karena developer game belum menyediakan terjemahan resmi Bahasa Indonesia di Steam.",
        "similarity-tooltip": "Skor ini dihitung menggunakan Cosine Similarity dari fitur game (genre, tag, deskripsi). Semakin tinggi persentasenya, semakin mirip dengan game referensimu.",
        "no-other-matches": "Tidak ada kecocokan tambahan",
        "gmodal-copy-btn": "Salin Link Game",
        "copied-text": "Tersalin!",
        "error-not-found": "Game '{game_name}' tidak ditemukan dalam sistem kami.",
        "footer-text": "LevelFind &copy; 2026 &mdash; Sistem Rekomendasi Game PC"
    },
    en: {
        "hero-title": "Start with a game<br>you <em>love</em>",
        "hero-sub": "Our AI analyzes thousands of titles to find the most accurate recommendations based on your favorite games.",
        "search-placeholder": "e.g., Elden Ring, Palworld...",
        "search-btn": "Discover",
        "history-label": "Recently Searched:",
        "suggestion-title": "Try one of these popular games instead:",
        "stat-games": "Games",
        "stat-algo": "Algorithm",
        "stat-sim": "Similarity",
        "analysis-result": "Analysis Results",
        "main-game": "Main Game",
        "game-profile": "Game Profile & Characteristics",
        "profile-sub": "The system analyzes and matches this game with others based on the following information:",
        "synopsis-label": "Synopsis / Game Description",
        "fallback_desc": "Detailed information for this game is not yet available in our system.",
        "synopsis-fallback": "A detailed synopsis for this game is not yet available in our system. The game is categorized by its features &amp; genres:",
        "desc-fallback": "Detailed information for this game is not yet available in our system.",
        "categories-label": "Category & Features (Tags & Genres)",
        "open-steam": "Open on Steam",
        "similar-rec": "Similar Recommendations",
        "share-btn": "Share",
        "genre-all": "All Genres",
        "price-all": "All Prices",
        "price-free": "Free to Play",
        "price-under10": "Under $10",
        "price-under30": "Under $30",
        "price-above30": "Over $30",
        "sort-match": "Highest Match",
        "sort-rating": "Best Rating",
        "sort-price": "Lowest Price",
        "no-search-title": "No search performed yet",
        "no-search-sub": "Start with one game you love — we will handle the rest.",
        "not-found-title": "Game Not Found",
        "not-found-sub": "Please check your spelling or try searching for another popular game.",
        "rank-1": "Your Reference Game",
        "read-more": "Read more",
        "show-less": "Show less",
        "indexed": "Indexed",
        "detail-btn": "View Details &#8594;",
        "loading-text": "Finding matches...",
        "modal-title": "How LevelFind Works",
        "modal-intro": "This system recommends games based on content similarity through 3 simple steps:",
        "modal-step1": "<strong>Data Cleaning:</strong> Cleans and organizes game descriptions, tags, and genres to make them machine-readable.",
        "modal-step2": "<strong>Keyword Weighting:</strong> Identifies unique keywords of a game that distinguish it most from other games.",
        "modal-step3": "<strong>Similarity Matching:</strong> Calculates how similar the target game keywords are to thousands of other games in our system to find the best match.",
        "modal-note": "*Match Percentage reflects how identical the story elements, mechanics, and atmosphere are compared to your search title.",
        "rating-na": "<i class=\"fas fa-minus-circle\" style=\"color: var(--text-3);\"></i> N/A (Not enough reviews yet)",
        "xai-title": "Recommendation Explanation (XAI)",
        "xai_title": "Recommendation Explanation (XAI)",
        "gmodal-features": "Matching Categories & Features:",
        "modal-desc-fallback": "Detailed information for this game is not yet available in our system.",
        "load-more-btn": "Load More",
        "similarity_label": "Similarity Level",
        "similarity_level": "Similarity Level",
        "top_contrib_title": "Top Matches",
        "other_shared_title": "Other Similarities",
        "calc_method_title": "Smart System",
        "calc_method_desc": "TF-IDF + Cosine Similarity",
        "view_xai_btn": "View XAI Explanation",
        "tight_cluster_badge": "Similar Scores",
        "synopsis-lang-notice-text": "This synopsis is displayed in English because the game developer has not provided an official Indonesian translation on Steam.",
        "similarity-tooltip": "This score is calculated using Cosine Similarity across game features (genre, tags, description). Higher percentage means more similar to your reference game.",
        "no-other-matches": "No additional matches found",
        "gmodal-copy-btn": "Copy Game Link",
        "copied-text": "Copied!",
        "error-not-found": "Game '{game_name}' was not found in our system.",
        "footer-text": "LevelFind &copy; 2026 &mdash; PC Game Recommendation System"
    }
};

window.generateXaiText = function(features, lang) {
    let isEn = (lang === 'en');
    
    if (!features || !Array.isArray(features) || features.length === 0) {
        return isEn 
            ? "This game is recommended based on overall key feature similarity."
            : "Game ini direkomendasikan berdasarkan tingkat kemiripan fitur utama.";
    }

    if (features.length === 1) {
        let f1 = features[0];
        let t1 = f1.type || (isEn ? 'Feature' : 'Fitur');
        let n1 = f1.name || '';
        return isEn
            ? `This game is recommended because it shares similar ${t1} ${n1} with your selected game.`
            : `Game ini direkomendasikan karena memiliki ${t1} ${n1} yang serupa dengan game yang Anda pilih.`;
    }

    let f1 = features[0];
    let f2 = features[1];
    let t1 = f1.type || (isEn ? 'Feature' : 'Fitur');
    let n1 = f1.name || '';
    let t2 = f2.type || (isEn ? 'Feature' : 'Fitur');
    let n2 = f2.name || '';

    if (isEn) {
        return `This game is recommended because it shares similar ${t1} ${n1} and ${t2} ${n2} with your selected game.`;
    } else {
        return `Game ini direkomendasikan karena memiliki ${t1} ${n1} dan ${t2} ${n2} yang serupa dengan game yang Anda pilih.`;
    }
};

function formatNumber(num, lang) {
    let n = typeof num === 'number' ? num : parseFloat(num);
    if (isNaN(n)) return num;
    return new Intl.NumberFormat(lang === 'id' ? 'id-ID' : 'en-US').format(n);
}

function updateDOMText(lang) {
    $('[data-num]').each(function() {
        let rawNum = $(this).attr('data-num');
        if (rawNum) {
            let formatted = formatNumber(rawNum, lang);
            if ($(this).hasClass('review-count-span')) {
                $(this).text(`(${formatted} reviews)`);
            } else {
                $(this).text(formatted);
            }
        }
    });

    $('[data-i18n]').each(function() {
        let key = $(this).attr('data-i18n');
        if (i18nDict[lang] && i18nDict[lang][key]) {
            $(this).html(i18nDict[lang][key]);
        }
    });

    $('[data-i18n-placeholder]').each(function() {
        let key = $(this).data('i18n-placeholder');
        if (i18nDict[lang] && i18nDict[lang][key]) {
            $(this).attr('placeholder', i18nDict[lang][key]);
        }
    });

    const $errorMsg = $('.error-msg');
    if ($errorMsg.length > 0) {
        let gameName = $errorMsg.attr('data-search-query') || '';
        let templateStr = i18nDict[lang]['error-not-found'];
        if (templateStr && gameName) {
            let translatedError = templateStr.replace('{game_name}', gameName);
            $errorMsg.find('.error-text').text(translatedError);
        }
    }

    $('.rec-card').each(function() {
        let card = $(this);
        let topFeatures = [];
        try {
            topFeatures = JSON.parse(card.attr('data-top-features') || '[]');
        } catch(err) {
            topFeatures = [];
        }

        if (topFeatures && topFeatures.length > 0) {
            let xaiText = generateXaiText(topFeatures, lang);
            card.find('.xai-text').text(xaiText);
            card.attr('data-dynamic-text', xaiText);
        } else {
            let expId = card.attr('data-exp-id');
            let expEn = card.attr('data-exp-en');
            if (lang === 'en' && expEn) {
                card.find('.xai-text').html(expEn);
            } else if (lang === 'id' && expId) {
                card.find('.xai-text').html(expId);
            }
        }
    });

    let remainingCards = $('.extra-card:hidden').length;
    if (remainingCards > 0) {
        let btnText = (lang === 'en') ? `Load More (${remainingCards} Games)` : `Tampilkan Lebih Banyak (${remainingCards} Game)`;
        $('[data-i18n="load-more-btn"]').text(btnText);
    }

    $('.badge-tight-cluster').each(function() {
        let titleText = (lang === 'en') ? $(this).data('title-en') : $(this).data('title-id');
        $(this).attr('title', titleText);
    });

    if (lang === 'id') {
        $('.spotlight-synopsis-notice, .spotlight-main-synopsis-notice').show();
        let modalDesc = $('#gmodal-desc').text().trim();
        let isFallback = $('#gmodal-desc').find('[data-i18n="fallback_desc"]').length > 0;
        if (modalDesc && modalDesc !== 'nan' && !isFallback) {
            $('.gmodal-synopsis-notice').show();
        } else {
            $('.gmodal-synopsis-notice').hide();
        }
    } else {
        $('.synopsis-lang-notice').hide();
    }

    let activeToggleBtn = $('.btn-toggle-other-features');
    if (activeToggleBtn.length > 0) {
        let extraContainer = activeToggleBtn.siblings('.xai-badge-group-extra');
        let textSpan = activeToggleBtn.find('.toggle-text');
        let hiddenCount = extraContainer.attr('data-hidden-count');
        if (extraContainer.is(':visible')) {
            textSpan.text(lang === 'en' ? 'Show Less' : 'Sembunyikan');
        } else {
            textSpan.text(lang === 'en' ? `Show All (${hiddenCount} more)` : `Tampilkan Semua (${hiddenCount} lainnya)`);
        }
    }
}

// XSS SANITIZATION HELPER FOR FRONTEND
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// 2. NATIVE VANILLA AUTOCOMPLETE DENGAN DEBOUNCE (280ms) & KEYBOARD NAVIGATION
function initNativeAutocomplete() {
    const gameInput = document.getElementById("game_input");
    if (!gameInput) return;

    const wrap = gameInput.closest(".search-wrap");
    if (!wrap) return;

    // Buat dropdown container jika belum ada (append ke document.body)
    let menu = document.querySelector(".custom-autocomplete-menu");
    if (!menu) {
        menu = document.createElement("ul");
        menu.className = "custom-autocomplete-menu";
        document.body.appendChild(menu);
    }

    let debounceTimer = null;
    let activeIndex = -1;
    let currentSuggestions = [];

    function positionMenu() {
        const rect = wrap.getBoundingClientRect();
        menu.style.position = "absolute";
        // Tambahkan offset 4px agar ada jarak rapi dari search bar
        menu.style.top = `${rect.bottom + window.scrollY + 4}px`;
        menu.style.left = `${rect.left + window.scrollX}px`;
        menu.style.width = `${rect.width}px`;
    }

    function hideMenu() {
        menu.classList.remove("show");
        menu.innerHTML = "";
        activeIndex = -1;
        currentSuggestions = [];
    }

    function renderSuggestions(items) {
        currentSuggestions = items;
        menu.innerHTML = "";
        activeIndex = -1;

        if (!items || items.length === 0) {
            hideMenu();
            return;
        }

        items.forEach((item, idx) => {
            const li = document.createElement("li");
            li.className = "custom-autocomplete-item";
            li.textContent = typeof item === 'object' ? item.label || item.value : item;
            li.addEventListener("click", () => {
                gameInput.value = li.textContent;
                hideMenu();
                gameInput.closest("form").submit();
            });
            menu.appendChild(li);
        });

        positionMenu();
        menu.classList.add("show");
    }

    function highlightItem(index) {
        const children = menu.querySelectorAll(".custom-autocomplete-item");
        children.forEach((child, i) => {
            if (i === index) {
                child.classList.add("active");
                child.scrollIntoView({ block: "nearest" });
            } else {
                child.classList.remove("active");
            }
        });
    }

    function fetchSuggestions(term) {
        fetch(`/api/search-suggestions?term=${encodeURIComponent(term)}`)
            .then(res => res.json())
            .then(data => {
                renderSuggestions(data);
            })
            .catch(err => {
                console.error("Autocomplete fetch error:", err);
                hideMenu();
            });
    }

    // Input Event listener dengan DEBOUNCE (280ms)
    gameInput.addEventListener("input", function() {
        clearTimeout(debounceTimer);
        const val = this.value.trim();

        if (val.length < 2) {
            hideMenu();
            return;
        }

        debounceTimer = setTimeout(() => {
            fetchSuggestions(val);
        }, 280);
    });

    // Keyboard Navigation (Panah Atas/Bawah, Enter, Escape)
    gameInput.addEventListener("keydown", function(e) {
        const items = menu.querySelectorAll(".custom-autocomplete-item");
        if (!menu.classList.contains("show") || items.length === 0) return;

        if (e.key === "ArrowDown") {
            e.preventDefault();
            activeIndex = (activeIndex + 1) % items.length;
            highlightItem(activeIndex);
        } else if (e.key === "ArrowUp") {
            e.preventDefault();
            activeIndex = (activeIndex - 1 + items.length) % items.length;
            highlightItem(activeIndex);
        } else if (e.key === "Enter") {
            if (activeIndex >= 0 && activeIndex < items.length) {
                e.preventDefault();
                gameInput.value = items[activeIndex].textContent;
                hideMenu();
                gameInput.closest("form").submit();
            }
        } else if (e.key === "Escape") {
            hideMenu();
        }
    });

    // Tutup autocomplete saat klik di luar area search input atau menu
    document.addEventListener("click", function(e) {
        if (!wrap.contains(e.target) && !menu.contains(e.target)) {
            hideMenu();
        }
    });

    // Reposisi saat resize window & tutup saat scroll
    window.addEventListener("resize", function() {
        if (menu.classList.contains("show")) {
            positionMenu();
        }
    });

    window.addEventListener("scroll", function() {
        if (menu.classList.contains("show")) {
            hideMenu();
        }
    }, { passive: true });
}

// MAIN DOM READY INITIALIZATION
$(document).ready(function() {
    const html = document.documentElement;

    // Set initial language from central state
    setLang(currentLang, false);

    // Inisialisasi Native Autocomplete Vanilla JS
    initNativeAutocomplete();

    // 1. THEME TOGGLE (DARK / LIGHT MODE WITH MORPH ANIMATION & LOCALSTORAGE)
    document.getElementById("theme-btn").addEventListener("click", function() {
        const btn = this;
        btn.classList.add("morphing");

        var isLight = html.getAttribute("data-theme") === "light";
        if (isLight) {
            html.removeAttribute("data-theme");
            localStorage.setItem("lf-theme", "dark");
        } else {
            html.setAttribute("data-theme", "light");
            localStorage.setItem("lf-theme", "light");
        }

        setTimeout(() => {
            btn.classList.remove("morphing");
        }, 300);
    });

    // 2. STAGGERED FADE-IN ANIMATION FOR RECOMMENDATION CARDS
    function triggerStaggeredFadeIn() {
        const $cards = $(".rec-grid .rec-card:visible");
        $cards.removeClass("show").addClass("stagger-anim");
        $cards.each(function(index) {
            const $card = $(this);
            setTimeout(function() {
                $card.addClass("show");
            }, index * 75);
        });
    }
    triggerStaggeredFadeIn();

    // 3. SEARCH HISTORY (LocalStorage - MAX 4 CHIPS UNTUK WRAPPING RAPI)
    function renderHistory() {
        let history = JSON.parse(localStorage.getItem('lf-history')) || [];
        let historyContainer = $("#search-history");
        historyContainer.find('.hist-chip').remove(); 

        if(history.length === 0) { historyContainer.hide(); return; }
        historyContainer.css("display", "flex");

        let visibleHistory = history.slice(0, 4);
        visibleHistory.forEach(item => {
            let safeTitle = escapeHtml(item);
            let chip = $(`<a href="/?q=${encodeURIComponent(item)}" class="tag hist-chip" title="${safeTitle}" style="background: transparent; cursor: pointer; border-color: var(--gold-dim); color: var(--gold); max-width: 220px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: inline-block; vertical-align: middle;">${safeTitle}</a>`);
            historyContainer.append(chip);
        });
    }
    renderHistory();

    // 4. SMOOTH LOADING STATE & SKELETON TRANSITION
    $('form').on('submit', function() {
        let query = $("#game_input").val().trim();
        if (query !== "") {
            let history = JSON.parse(localStorage.getItem('lf-history')) || [];
            history = history.filter(item => item.toLowerCase() !== query.toLowerCase()); 
            history.unshift(query); 
            if(history.length > 5) history.pop(); 
            localStorage.setItem('lf-history', JSON.stringify(history));

            let topNVal = parseInt($("#top_n_select").val()) || 8;

            $(".container").fadeOut(180, function() {
                let skeletonCardHtml = `
                    <div class="skeleton-card-wrapper" style="background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--r-md); overflow: hidden; padding: 0.9rem; display: flex; flex-direction: column; gap: 10px; height: 320px;">
                        <div class="skeleton" style="width: 100%; height: 120px; border-radius: var(--r-sm);"></div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div class="skeleton" style="width: 60%; height: 16px;"></div>
                            <div class="skeleton" style="width: 25%; height: 16px;"></div>
                        </div>
                        <div class="skeleton" style="width: 85%; height: 14px;"></div>
                        <div class="skeleton" style="width: 100%; height: 40px; margin-top: 6px;"></div>
                        <div class="skeleton" style="width: 100%; height: 32px; margin-top: auto; border-radius: var(--r-sm);"></div>
                    </div>
                `;
                let skeletonHtml = `
                    <div class="skeleton skeleton-spotlight" style="height: 200px; margin-bottom: 2rem; border-radius: var(--r-lg);"></div>
                    <div class="section-header" style="margin-bottom: 1.5rem;"><span class="skeleton" style="width: 180px; height: 24px;"></span></div>
                    <div class="rec-grid">
                        ${Array(topNVal).fill(skeletonCardHtml).join('')}
                    </div>
                `;
                $(this).html(skeletonHtml).fadeIn(250);
            });

            $("#loading-overlay").css("display", "flex").hide().fadeIn(250);
        }
    });

    // 5. CLIENT-SIDE SORTING
    $("#sort-grid").on("change", function() {
        let sortBy = $(this).val();
        let grid = $(".rec-grid");
        let cards = grid.children(".rec-card").get();

        cards.sort(function(a, b) {
            let valA, valB;
            if(sortBy === "price_asc") {
                valA = parseFloat($(a).data("price")); valB = parseFloat($(b).data("price"));
                return valA - valB; 
            } else if(sortBy === "rating_desc") {
                valA = parseFloat($(a).data("rating")); valB = parseFloat($(b).data("rating"));
                return valB - valA; 
            } else {
                valA = parseFloat($(a).data("match")); valB = parseFloat($(b).data("match"));
                return valB - valA; 
            }
        });
        $.each(cards, function(index, card) { grid.append(card); }); 
        triggerStaggeredFadeIn();
    });

    // 5.1 CLIENT-SIDE FILTERING (GENRE & PRICE)
    function applyFilters() {
        let genreVal = $("#filter-genre").val() ? $("#filter-genre").val().toLowerCase() : "all";
        let priceVal = $("#filter-price").val() || "all";
        let visibleCount = 0;

        $(".rec-card").each(function() {
            let cardGenres = ($(this).attr("data-genres") || "").toLowerCase();
            let cardPrice = parseFloat($(this).attr("data-price")) || 0;

            let matchGenre = (genreVal === "all") || cardGenres.includes(genreVal);
            let matchPrice = false;

            if (priceVal === "all") {
                matchPrice = true;
            } else if (priceVal === "free") {
                matchPrice = (cardPrice === 0);
            } else if (priceVal === "under10") {
                matchPrice = (cardPrice > 0 && cardPrice <= 10);
            } else if (priceVal === "under30") {
                matchPrice = (cardPrice > 0 && cardPrice <= 30);
            } else if (priceVal === "above30") {
                matchPrice = (cardPrice > 30);
            }

            if (matchGenre && matchPrice) {
                $(this).fadeIn(200);
                visibleCount++;
            } else {
                $(this).hide();
            }
        });

        triggerStaggeredFadeIn();

        if (visibleCount === 0) {
            if ($("#empty-filter-msg").length === 0) {
                $(".rec-grid").after('<div id="empty-filter-msg" style="text-align: center; padding: 2rem; color: var(--text-2); background: var(--bg-elevated); border-radius: var(--r-md); border: 1px dashed var(--border-md); margin-top: 1rem;"><i class="fas fa-filter" style="font-size: 2rem; color: var(--text-3); margin-bottom: 0.5rem; display: block;"></i> Tidak ada game rekomendasi yang cocok dengan kombinasi filter ini.</div>');
            } else {
                $("#empty-filter-msg").show();
            }
        } else {
            $("#empty-filter-msg").hide();
        }
    }

    $("#filter-genre, #filter-price").on("change", applyFilters);

    // 6. SHARE LINK (COPY CLIPBOARD)
    $("#share-btn").on("click", function() {
        let actualTitleAttr = $(this).attr("data-share-title") || "";
        let currentUrl = window.location.origin + "/?q=" + encodeURIComponent(actualTitleAttr);
        let btn = $(this);
        let copiedText = (i18nDict[currentLang] && i18nDict[currentLang]['copied-text']) ? i18nDict[currentLang]['copied-text'] : 'Tersalin!';
        let originalHtml = btn.html();

        navigator.clipboard.writeText(currentUrl).then(() => {
            btn.html(`<i class="fas fa-check"></i> <span>${copiedText}</span>`);
            btn.css({"background": "var(--gold)", "color": "#000"});
            setTimeout(() => { 
                btn.html(originalHtml); 
                btn.css({"background": "transparent", "color": "var(--gold)"});
            }, 1500);
        });
    });

    // 7. MODAL XAI LOGIC
    const modal = document.getElementById('ai-modal');
    const infoBtn = document.getElementById('info-btn');
    const closeBtn = document.getElementById('close-modal');

    if (infoBtn) {
        infoBtn.addEventListener('click', function() {
            modal.classList.add('show');
            modal.setAttribute('aria-hidden', 'false');
        });
    }

    if (closeBtn) {
        closeBtn.addEventListener('click', function() {
            modal.classList.remove('show');
            modal.setAttribute('aria-hidden', 'true');
        });
    }

    window.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.classList.remove('show');
            modal.setAttribute('aria-hidden', 'true');
        }
    });

    // 8. TOGGLE ACCORDION BOBOT AI
    $("#toggle-bobot").on("click", function() {
        $("#content-bobot").slideToggle(300);
        $(this).find(".toggle-icon").toggleClass("rotate");
    });

    // LANGUAGE TOGGLE EVENT
    $('#lang-btn').on('click', function() {
        let nextLang = (currentLang === 'id') ? 'en' : 'id';
        setLang(nextLang, true);
    });

    // 10. GAME DETAIL MODAL
    const gmodal = document.getElementById('game-detail-modal');
    const closeGModal = document.getElementById('close-gmodal');

    function renderFeatureList(featuresObj, isTop = true) {
        let emptyMsgKey = 'no-other-matches';
        let emptyMsg = (i18nDict[currentLang] && i18nDict[currentLang][emptyMsgKey]) ? i18nDict[currentLang][emptyMsgKey] : 'Tidak ada kecocokan tambahan';
        let emptyHtml = `<div style="color: var(--text-3); font-style: italic; font-size: 0.78rem;" data-i18n="${emptyMsgKey}">${emptyMsg}</div>`;

        if (!featuresObj) return emptyHtml;
        let items = [];

        if (featuresObj.genres && Array.isArray(featuresObj.genres)) {
            featuresObj.genres.forEach(g => {
                let name = (typeof g === 'object' && g !== null) ? g.name : g;
                let pct = (typeof g === 'object' && g !== null && g.pct !== undefined) ? g.pct : null;
                let score = (typeof g === 'object' && g !== null && g.score !== undefined) ? g.score : 0;
                items.push({ type: 'Genre', name: name, pct: pct, score: score });
            });
        }
        if (featuresObj.tags && Array.isArray(featuresObj.tags)) {
            featuresObj.tags.forEach(t => {
                let name = (typeof t === 'object' && t !== null) ? t.name : t;
                let pct = (typeof t === 'object' && t !== null && t.pct !== undefined) ? t.pct : null;
                let score = (typeof t === 'object' && t !== null && t.score !== undefined) ? t.score : 0;
                items.push({ type: 'Tag', name: name, pct: pct, score: score });
            });
        }

        if (items.length === 0) return emptyHtml;

        items.sort((a, b) => (b.pct || b.score || 0) - (a.pct || a.score || 0));

        function buildBadgeHtml(item) {
            let label = item.type === 'Genre' ? `Genre: ${escapeHtml(item.name)}` : escapeHtml(item.name);
            let hasPct = (item.pct !== null && item.pct !== undefined && item.pct > 0);
            let badgeClass = hasPct ? 'xai-badge-with-score' : 'xai-badge-no-score';
            let pctText = hasPct 
                ? ` <span class="badge-pct" style="opacity: 0.9; font-size: 0.85em; font-weight: 600; margin-left: 4px;">· ${item.pct}%</span>` 
                : '';
            return `<span class="${badgeClass}">${label}${pctText}</span>`;
        }

        if (isTop || items.length <= 4) {
            let html = '<div class="xai-badge-group" style="display: flex; flex-wrap: wrap; gap: 6px;">';
            items.forEach(item => {
                html += buildBadgeHtml(item);
            });
            html += '</div>';
            return html;
        } else {
            let initialItems = items.slice(0, 4);
            let hiddenItems = items.slice(4);
            let hiddenCount = hiddenItems.length;

            let html = '<div class="xai-badge-group" style="display: flex; flex-wrap: wrap; gap: 6px;">';
            initialItems.forEach(item => {
                html += buildBadgeHtml(item);
            });
            html += '</div>';

            html += `<div class="xai-badge-group-extra" style="display: none; margin-top: 6px;" data-hidden-count="${hiddenCount}">`;
            html += '<div class="xai-badge-group" style="display: flex; flex-wrap: wrap; gap: 6px;">';
            hiddenItems.forEach(item => {
                html += buildBadgeHtml(item);
            });
            html += '</div>';
            html += '</div>';

            let btnText = (currentLang === 'en') ? `Show All (${hiddenCount} more)` : `Tampilkan Semua (${hiddenCount} lainnya)`;
            html += `<button type="button" class="btn-toggle-other-features" style="background: none; border: none; color: var(--gold); font-size: 0.75rem; padding: 4px 0 0 0; margin-top: 4px; cursor: pointer; display: inline-flex; align-items: center; gap: 5px; font-weight: 500; transition: color 0.2s ease;">
                <span class="toggle-text">${btnText}</span>
                <i class="fas fa-chevron-down" style="font-size: 0.65rem; transition: transform 0.25s ease-out;"></i>
            </button>`;

            return html;
        }
    }

    $(document).on('click', '.btn-toggle-other-features', function(e) {
        e.preventDefault();
        let btn = $(this);
        let extraContainer = btn.siblings('.xai-badge-group-extra');
        let chevron = btn.find('i');
        let textSpan = btn.find('.toggle-text');
        let hiddenCount = extraContainer.attr('data-hidden-count');

        if (extraContainer.is(':visible')) {
            extraContainer.slideUp(220);
            chevron.css('transform', 'rotate(0deg)');
            let showText = (currentLang === 'en') ? `Show All (${hiddenCount} more)` : `Tampilkan Semua (${hiddenCount} lainnya)`;
            textSpan.text(showText);
        } else {
            extraContainer.slideDown(220);
            chevron.css('transform', 'rotate(180deg)');
            let hideText = (currentLang === 'en') ? `Show Less` : `Sembunyikan`;
            textSpan.text(hideText);
        }
    });

    $(document).on('click', '.btn-open-game-modal', function(e) {
        e.preventDefault();
        $('.rec-card').removeClass('active-modal-card');
        let card = $(this).closest('.rec-card');
        card.addClass('active-modal-card');

        let name = card.attr('data-name');
        let appid = card.attr('data-appid');
        let image = card.attr('data-image');
        let simPct = card.attr('data-similarity-pct') || card.attr('data-match-val') || '90';
        let cosineScore = card.attr('data-cosine-score') || card.attr('data-score') || '0.90';

        let xaiId = card.attr('data-xai-id') || card.attr('data-exp-id') || '';
        let xaiEn = card.attr('data-xai-en') || card.attr('data-exp-en') || '';

        let topFeatures = {};
        let otherFeatures = {};
        try {
            topFeatures = JSON.parse(card.attr('data-top-features') || '{}');
        } catch(err) { topFeatures = {}; }
        try {
            otherFeatures = JSON.parse(card.attr('data-other-features') || '{}');
        } catch(err) { otherFeatures = {}; }

        let desc = card.attr('data-desc') || '';

        updateDOMText(currentLang);

        let matchInfo = getMatchTier(simPct);

        $('#gmodal-img').attr('src', image);
        $('#gmodal-title').text(name);
        $('#gmodal-match').text(simPct + '%').attr('class', 'match-badge ' + matchInfo.badgeClass).css({'border-color': matchInfo.color, 'color': matchInfo.color});

        $('#gmodal-match-pct').text(simPct + '%');
        $('#gmodal-cosine-score').text(`(Cosine: ${cosineScore})`);

        let activeXaiText = (currentLang === 'en') ? (xaiEn || xaiId) : (xaiId || xaiEn);
        $('#gmodal-xai').html(escapeHtml(activeXaiText));

        let isSparse = card.attr('data-is-sparse') === 'true';
        if (isSparse) {
            let sparseMsg = (currentLang === 'en')
                ? 'This similarity score is based only on genre & tag data, as a full description for this game is not yet available in our database, so accuracy may be more limited compared to games with full descriptions.'
                : 'Skor kemiripan ini dihitung hanya dari genre & tag karena deskripsi lengkap game ini belum tersedia di database kami, sehingga tingkat akurasinya mungkin lebih terbatas dibanding game dengan deskripsi lengkap.';
            $('#gmodal-sparse-notice-text').text(sparseMsg);
            $('#gmodal-sparse-notice').show();
        } else {
            $('#gmodal-sparse-notice').hide();
        }

        $('#gmodal-top-features').html(renderFeatureList(topFeatures, true));
        $('#gmodal-other-features').html(renderFeatureList(otherFeatures, false));

        let fallbackText = (i18nDict[currentLang] && i18nDict[currentLang]['fallback_desc']) ? i18nDict[currentLang]['fallback_desc'] : 'Informasi detail untuk game ini belum tersedia di sistem kami.';
        if (!desc || desc.trim() === '' || desc === 'nan') {
            $('#gmodal-desc').html(`<span data-i18n="fallback_desc">${fallbackText}</span>`);
            $('.gmodal-synopsis-notice').hide();
        } else {
            $('#gmodal-desc').text(desc);
            if (currentLang === 'id') {
                $('.gmodal-synopsis-notice').show();
            } else {
                $('.gmodal-synopsis-notice').hide();
            }
        }
        $('#gmodal-steam').attr('href', 'https://store.steampowered.com/app/' + encodeURIComponent(appid));

        $('#gmodal-copy-link').attr('data-game-title', name);

        if (gmodal) {
            gmodal.classList.add('show');
            gmodal.setAttribute('aria-hidden', 'false');
            document.body.style.overflow = 'hidden';

            const scrollContainer = document.getElementById('gmodal-scroll-container');
            if (scrollContainer) {
                scrollContainer.scrollTop = 0;
                checkGmodalScrollPosition(scrollContainer);
            }

            animateSimilarityScore(simPct);
            staggerFeatureBadges();
        }
    });

    function animateSimilarityScore(targetPctStr) {
        let targetVal = parseFloat(targetPctStr) || 0;
        let isReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        let $pctEl = $('#gmodal-match-pct');
        let $barEl = $('#gmodal-progress-bar');

        if (isReducedMotion) {
            $pctEl.text(targetVal.toFixed(1) + '%');
            $barEl.css('width', Math.min(targetVal, 100) + '%');
            return;
        }

        let startVal = 0;
        let duration = 600;
        let startTime = null;

        $pctEl.text('0.0%');
        $barEl.css('width', '0%');

        function step(timestamp) {
            if (!startTime) startTime = timestamp;
            let progress = Math.min((timestamp - startTime) / duration, 1);
            let currentVal = startVal + progress * (targetVal - startVal);

            $pctEl.text(currentVal.toFixed(1) + '%');
            $barEl.css('width', Math.min(currentVal, 100) + '%');

            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                $pctEl.text(targetVal.toFixed(1) + '%');
                $barEl.css('width', Math.min(targetVal, 100) + '%');
            }
        }

        requestAnimationFrame(step);
    }

    function staggerFeatureBadges() {
        let isReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (isReducedMotion) return;

        $('#gmodal-top-features .xai-badge-with-score, #gmodal-top-features .xai-badge-no-score').each(function(idx) {
            $(this).css('animation-delay', (idx * 85) + 'ms');
        });
        $('#gmodal-other-features .xai-badge-with-score, #gmodal-other-features .xai-badge-no-score').each(function(idx) {
            $(this).css('animation-delay', ((idx + 2) * 85) + 'ms');
        });
    }

    function checkGmodalScrollPosition(el) {
        if (!el) return;
        const isScrollable = el.scrollHeight > el.clientHeight + 5;
        const isAtBottom = (el.scrollTop + el.clientHeight) >= (el.scrollHeight - 10);

        if (!isScrollable) {
            $(el).addClass('no-scroll').removeClass('at-bottom');
        } else if (isAtBottom) {
            $(el).addClass('at-bottom').removeClass('no-scroll');
        } else {
            $(el).removeClass('at-bottom no-scroll');
        }
    }

    $('#gmodal-scroll-container').on('scroll', function() {
        checkGmodalScrollPosition(this);
    });

    $(document).on('click', '#gmodal-copy-link', function(e) {
        e.preventDefault();
        let gameTitle = $(this).attr('data-game-title') || $('#gmodal-title').text().trim();
        let currentUrl = window.location.origin + "/?q=" + encodeURIComponent(gameTitle);

        let btn = $(this);
        let copiedText = (i18nDict[currentLang] && i18nDict[currentLang]['copied-text']) ? i18nDict[currentLang]['copied-text'] : 'Tersalin!';
        let originalBtnText = (i18nDict[currentLang] && i18nDict[currentLang]['gmodal-copy-btn']) ? i18nDict[currentLang]['gmodal-copy-btn'] : 'Salin Link Game';

        navigator.clipboard.writeText(currentUrl).then(() => {
            btn.html(`<i class="fas fa-check" style="transform: scale(1.1); transition: transform 0.2s ease;"></i> <span>${copiedText}</span>`);
            btn.css({"background": "var(--gold)", "color": "#000", "border-color": "var(--gold)"});
            setTimeout(() => { 
                btn.html(`<i class="fas fa-link"></i> <span data-i18n="gmodal-copy-btn">${originalBtnText}</span>`); 
                btn.css({"background": "transparent", "color": "var(--gold)", "border-color": "var(--gold)"});
            }, 1500);
        });
    });

    function initScrollReveal() {
        const elements = document.querySelectorAll('.spotlight, .rec-card, .section-header, .hero-deco');
        elements.forEach(el => el.classList.add('reveal-on-scroll'));

        if ('IntersectionObserver' in window) {
            const observer = new IntersectionObserver((entries, obs) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('is-visible');
                        obs.unobserve(entry.target);
                    }
                });
            }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

            elements.forEach(el => observer.observe(el));
        } else {
            elements.forEach(el => el.classList.add('is-visible'));
        }
    }

    initScrollReveal();

    $(document).on('click', '.custom-tooltip-wrapper', function(e) {
        e.stopPropagation();
        $('.custom-tooltip-wrapper').not(this).removeClass('tooltip-active');
        $(this).toggleClass('tooltip-active');
    });

    $(document).on('click', function() {
        $('.custom-tooltip-wrapper').removeClass('tooltip-active');
    });

    function closeGameModal() {
        if (gmodal) {
            gmodal.classList.remove('show');
            gmodal.setAttribute('aria-hidden', 'true');
        }
        document.body.style.overflow = '';
        $('.rec-card').removeClass('active-modal-card');
    }

    if (closeGModal) {
        closeGModal.addEventListener('click', closeGameModal);
    }

    window.addEventListener('click', function(e) {
        if (e.target === gmodal) {
            closeGameModal();
        }
    });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && gmodal && gmodal.classList.contains('show')) {
            closeGameModal();
        }
    });

    $(document).on('click', '#load-more-btn', function() {
        $('.extra-card').fadeIn(400);
        $('#load-more-container').fadeOut(300);
    });

    $(window).on('scroll', function() {
        if ($(this).scrollTop() > 250) {
            $('#scroll-top-btn').addClass('show');
        } else {
            $('#scroll-top-btn').removeClass('show');
        }
    });

    $('#scroll-top-btn').on('click', function() {
        $('html, body').animate({ scrollTop: 0 }, 400);
    });

    $('.match-badge').each(function() {
        let $this = $(this);
        let text = $this.text().trim();
        let match = text.match(/(\d+)/);
        if (match) {
            let targetVal = parseInt(match[1]);
            $({ countNum: 0 }).animate({ countNum: targetVal }, {
                duration: 1000,
                easing: 'swing',
                step: function() {
                    let simText = (i18nDict[currentLang] && i18nDict[currentLang]['similarity_label']) ? i18nDict[currentLang]['similarity_label'] : 'Tingkat Kemiripan';
                    $this.html(`<span class="match-val">${Math.floor(this.countNum)}</span>% <span class="similarity-text" data-i18n="similarity_label">${simText}</span>`);
                },
                complete: function() {
                    let simText = (i18nDict[currentLang] && i18nDict[currentLang]['similarity_label']) ? i18nDict[currentLang]['similarity_label'] : 'Tingkat Kemiripan';
                    $this.html(`<span class="match-val">${targetVal}</span>% <span class="similarity-text" data-i18n="similarity_label">${simText}</span>`);
                }
            });
        }
    });

    function checkSpotlightTruncation() {
        const $desc = $('#spotlight-desc-text');
        const $btn = $('#btn-toggle-spotlight-desc');
        if ($desc.length === 0) return;

        let el = $desc[0];
        if ($desc.hasClass('expanded')) return;

        if (el.scrollHeight > el.clientHeight + 4) {
            $btn.css('display', 'inline-flex');
        } else {
            $btn.hide();
        }
    }

    checkSpotlightTruncation();
    $(window).on('load resize', checkSpotlightTruncation);
    if (document.fonts && document.fonts.ready) {
        document.fonts.ready.then(checkSpotlightTruncation);
    }

    $('#btn-toggle-spotlight-desc').on('click', function() {
        const $desc = $('#spotlight-desc-text');
        const $btn = $(this);

        $desc.toggleClass('expanded');
        const isExpanded = $desc.hasClass('expanded');

        let readMoreText = (i18nDict[currentLang] && i18nDict[currentLang]['read-more']) ? i18nDict[currentLang]['read-more'] : 'Baca selengkapnya';
        let showLessText = (i18nDict[currentLang] && i18nDict[currentLang]['show-less']) ? i18nDict[currentLang]['show-less'] : 'Sembunyikan';

        if (isExpanded) {
            $btn.html(`<span data-i18n="show-less">${showLessText}</span> <i class="fas fa-chevron-up" style="font-size: 0.7em;"></i>`);
        } else {
            $btn.html(`<span data-i18n="read-more">${readMoreText}</span> <i class="fas fa-chevron-down" style="font-size: 0.7em;"></i>`);
        }
    });
});
