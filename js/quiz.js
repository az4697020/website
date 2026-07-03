(() => {
    const params = new URLSearchParams(window.location.search);
    const qType = params.get('type') || 'single';

    const typeLabels = {
        single: '单选题',
        multi: '多选题',
        judge: '判断题'
    };

    const loadingEl = document.getElementById('loading');
    const errorEl = document.getElementById('error');
    const bankMainEl = document.getElementById('bank-main');
    const pageTitleEl = document.getElementById('page-title');
    const questionsListEl = document.getElementById('questions-list');

    function init() {
        const label = typeLabels[qType] || '题库';
        document.title = `${label} - 人工智能训练师（三级）`;
        pageTitleEl.textContent = label;

        fetch('data/questions.json')
            .then(res => {
                if (!res.ok) throw new Error('Network response was not ok');
                return res.json();
            })
            .then(data => {
                const questions = data[qType] || [];
                renderQuestions(questions);
                loadingEl.classList.add('hidden');
                bankMainEl.classList.remove('hidden');
            })
            .catch(err => {
                console.error(err);
                loadingEl.classList.add('hidden');
                errorEl.classList.remove('hidden');
            });
    }

    function renderQuestions(questions) {
        questionsListEl.innerHTML = '';
        questions.forEach(q => {
            const card = document.createElement('div');
            card.className = 'bank-card';

            const header = document.createElement('div');
            header.className = 'question-header';

            const number = document.createElement('span');
            number.className = 'bank-number';
            number.textContent = `第 ${q.id} 题`;

            const text = document.createElement('span');
            text.className = 'question-text';
            text.textContent = q.question;

            header.appendChild(number);
            header.appendChild(text);

            const options = document.createElement('div');
            options.className = 'options-list';
            q.options.forEach(opt => {
                const isCorrect = Array.isArray(q.answer)
                    ? q.answer.includes(opt.key)
                    : q.answer === opt.key;
                const item = document.createElement('div');
                item.className = 'option-item' + (isCorrect ? ' correct' : '');
                item.innerHTML = `
                    <span class="option-label">${opt.key}.</span>
                    <span class="option-text">${escapeHtml(opt.text)}</span>
                `;
                options.appendChild(item);
            });

            card.appendChild(header);
            card.appendChild(options);
            questionsListEl.appendChild(card);
        });
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // 回到顶部按钮
    function initBackToTop() {
        const btn = document.createElement('button');
        btn.className = 'btn-back-to-top';
        btn.innerHTML = '&#8593;';
        btn.title = '回到顶部';
        btn.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
        document.body.appendChild(btn);

        let ticking = false;
        window.addEventListener('scroll', () => {
            if (!ticking) {
                requestAnimationFrame(() => {
                    if (window.scrollY > 400) {
                        btn.classList.add('visible');
                    } else {
                        btn.classList.remove('visible');
                    }
                    ticking = false;
                });
                ticking = true;
            }
        }, { passive: true });
    }

    init();
    initBackToTop();
})();
