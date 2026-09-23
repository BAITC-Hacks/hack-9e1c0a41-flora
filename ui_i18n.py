"""Offline UI translations. Agent output and exported data keep their originals."""

import re

import streamlit as st

LANGUAGES = {"ru": "Русский", "kk": "Қазақша", "en": "English"}

# Russian source text, English, Kazakh. Placeholders are identical across locales.
TEXT = {
    "Ожидаемый чистый прирост": ("Expected net gain", "Күтілетін таза өсім"),
    "Сегмент не указан": ("Segment unspecified", "Сегмент көрсетілмеген"),
    "Технический ID: {name}": ("Technical ID: {name}", "Техникалық ID: {name}"),
    "Ожидаемый чистый прирост и нижняя оценка взяты из модели агента. Фактический результат скорера показан в разделе «Результат».": ("Expected net gain and the lower estimate come from the agent model. The actual scored result is shown under Results.", "Күтілетін таза өсім мен төменгі баға агент моделінен алынған. Нақты бағалау нәтижесі «Нәтиже» бөлімінде көрсетілген."),
    "{n} абонентов · бюджет {budget} · до {campaigns} кампаний": ("{n} subscribers · {budget} budget · up to {campaigns} campaigns", "{n} абонент · бюджет {budget} · {campaigns} науқанға дейін"),
    "Модель": ("Model", "Модель"),
    "Обучение": ("Learning", "Үйрену"),
    "Итоговый план": ("Final plan", "Соңғы жоспар"),
    "Ожидает запуска": ("Ready to run", "Іске қосуға дайын"),
    "Результат": ("Results", "Нәтиже"),
    "Решения": ("Decisions", "Шешімдер"),
    "План": ("Plan", "Жоспар"),
    "Устойчивость": ("Robustness", "Тұрақтылық"),
    "Объяснение": ("Explanation", "Түсіндірме"),
    "Раздел": ("Section", "Бөлім"),
    "ТАРИФНЫЕ КАМПАНИИ": ("TARIFF CAMPAIGNS", "ТАРИФТІК НАУҚАНДАР"),
    "Тарифные кампании": ("Tariff campaigns", "Тарифтік науқандар"),
    "Как думает агент": ("Agent decisions", "Агент шешімдері"),
    "План кампаний": ("Campaign plan", "Науқан жоспары"),
    "Устойчивость стратегии": ("Strategy robustness", "Стратегия тұрақтылығы"),
    "Объяснение решения": ("Decision rationale", "Шешім негіздемесі"),
    "Рабочее пространство": ("Workspace", "Жұмыс кеңістігі"),
    "Локальная среда": ("Local environment", "Жергілікті орта"),
    "Команда Flora": ("Team Flora", "Flora командасы"),
    "Точные решения. Бережный рост.": ("Precise decisions. Thoughtful growth.", "Дәл шешімдер. Байыпты өсу."),
    "Розовые цветы и зелёные листья": ("Pink flowers and green leaves", "Қызғылт гүлдер мен жасыл жапырақтар"),
    "Локально · без API-ключей": ("Local · no API keys", "Жергілікті · API кілтінсіз"),
    "Тема {current}. Включить {target}": ("{current} theme. Switch to {target}", "{current} тақырыбы. {target} тақырыбына ауысу"),
    "Рассчитать кампании": ("Run campaigns", "Есептеу"),
    "Агент проводит пилоты и формирует план…": ("Running pilots and building the plan…", "Пилоттар орындалып, жоспар құрылуда…"),
    "Seed {seed} · {time} · результат сохранён": ("Seed {seed} · {time} · result saved", "Seed {seed} · {time} · нәтиже сақталды"),
    "Локальная мок-среда · расчёт не запущен": ("Local mock environment · not run yet", "Жергілікті сынақ ортасы · есептеу басталмады"),
    "Код или данные изменились. Показан предыдущий расчёт.": ("Code or data changed. Showing the previous run.", "Код немесе деректер өзгерді. Алдыңғы есеп көрсетілген."),
    "Показан результат для seed {shown}; выбранный seed {selected} ещё не рассчитан.": ("Showing seed {shown}; selected seed {selected} has not been run yet.", "Seed {shown} нәтижесі көрсетілген; таңдалған seed {selected} әлі есептелмеді."),
    "Файлы проекта недоступны: {error}": ("Project files unavailable: {error}", "Жоба файлдары қолжетімсіз: {error}"),
    "Данные организаторов Beeline · Симуляция, не реальные рассылки": ("Beeline organizer data · simulation, no real messages", "Beeline ұйымдастырушыларының деректері · симуляция, нақты хабарлама жіберілмейді"),
    "ОЖИДАНИЕ РАСЧЁТА": ("AWAITING RESULTS", "НӘТИЖЕНІ КҮТУ"),
    "ИТОГ СИМУЛЯЦИИ": ("SIMULATION RESULTS", "СИМУЛЯЦИЯ НӘТИЖЕСІ"),
    "Суммы в условных единицах": ("Amounts in simulation units", "Сомалар шартты бірліктермен"),
    "Чистый результат": ("Net result", "Таза нәтиже"),
    "Затраты": ("Cost", "Шығын"),
    "Уникальный охват": ("Unique reach", "Бірегей қамту"),
    "Пилоты": ("Pilots", "Пилоттар"),
    "Кампании": ("Campaigns", "Науқандар"),
    "После стоимости контактов": ("After contact costs", "Байланыс шығынын шегергенде"),
    "{value} к шаблону": ("{value} vs. baseline", "Үлгімен салыстырғанда {value}"),
    "из бюджета 100 000": ("of a 100,000 budget", "100 000 бюджеттен"),
    "абонентов": ("subscribers", "абонент"),
    "из 20 доступных": ("of 20 available", "қолжетімді 20-дан"),
    "из 10 доступных": ("of 10 available", "қолжетімді 10-нан"),
    "{value}% лимита": ("{value}% of limit", "лимиттің {value}%-ы"),
    "Каждый контакт стоит денег. Flora проверяет, какие переходы окупаются.": ("Every contact has a cost. Flora tests which tariff changes pay off.", "Әр байланыс шығын талап етеді. Flora қай тариф ауысуы тиімді екенін тексереді."),
    "Обучение меняет результат": ("Learning changes the outcome", "Үйрену нәтижені өзгертеді"),
    "Чистый прирост ARPU · млн условных единиц": ("Net ARPU gain · million simulation units", "ARPU таза өсімі · млн шартты бірлік"),
    "Результаты ещё не получены": ("No results yet", "Нәтиже әлі жоқ"),
    "Локальная симуляция · одинаковый seed для двух агентов": ("Local simulation · the same seed for both agents", "Жергілікті симуляция · екі агентке бірдей seed"),
    "Шаблон": ("Baseline", "Үлгі"),
    "Агент": ("Agent", "Агент"),
    "Чистый результат, условные единицы": ("Net result, simulation units", "Таза нәтиже, шартты бірлік"),
    "млн условных единиц": ("million simulation units", "млн шартты бірлік"),
    "млн": ("M", "млн"),
    "Ресурсы кампании": ("Campaign resources", "Науқан ресурстары"),
    "Общие лимиты пилотов и финального плана": ("Shared limits for pilots and the final plan", "Пилоттар мен соңғы жоспардың ортақ лимиттері"),
    "Бюджет": ("Budget", "Бюджет"),
    "Контакты": ("Contacts", "Байланыстар"),
    "Сравнение стратегий": ("Strategy comparison", "Стратегияларды салыстыру"),
    "Условные единицы симулятора, не евро и не тенге.": ("Simulation units, not euros or tenge.", "Симулятордың шартты бірліктері, еуро немесе теңге емес."),
    "Условные единицы симулятора.": ("Simulation units.", "Симулятордың шартты бірліктері."),
    "Время, с": ("Time, s", "Уақыт, с"),
    "Ошибка": ("Error", "Қате"),
    "Мок-модель. Результат включает пилоты, стоимость контактов и дедупликацию. Это не прогноз прибыли Beeline.": ("Mock model. Includes pilots, contact costs and deduplication. This is not a Beeline profit forecast.", "Сынақ моделі. Нәтижеге пилоттар, байланыс шығыны және қайталануды жою кіреді. Бұл Beeline пайдасының болжамы емес."),
    "Журнал пилотного обучения": ("Pilot learning log", "Пилоттық үйрену журналы"),
    "Пилоты ещё не проведены": ("No pilots run yet", "Пилоттар әлі орындалмады"),
    "Гипотеза → наблюдение → обновлённая оценка": ("Hypothesis → observation → updated estimate", "Гипотеза → бақылау → жаңартылған баға"),
    "В журнале нет выполненных пилотов.": ("No completed pilots in the log.", "Журналда орындалған пилоттар жоқ."),
    "Шаг": ("Step", "Қадам"),
    "Шаг {step}": ("Step {step}", "{step}-қадам"),
    "Текущий тариф": ("Current tariff", "Ағымдағы тариф"),
    "Целевой тариф": ("Target tariff", "Мақсатты тариф"),
    "Канал": ("Channel", "Арна"),
    "Абонентов": ("Subscribers", "Абоненттер"),
    "Пилот, %": ("Observed lift, %", "Бақыланған өсім, %"),
    "До, ±2σ": ("Before, ±2σ", "Дейін, ±2σ"),
    "После, ±2σ": ("After, ±2σ", "Кейін, ±2σ"),
    "Решение агента": ("Agent rationale", "Агент негіздемесі"),
    "Пилот": ("Pilot", "Пилот"),
    "нет в журнале": ("not logged", "журналда жоқ"),
    "После: оценка ± 2 стандартных отклонения модели. До: значения из журнала; это не гарантия прибыли и не одновременный доверительный интервал всех гипотез.": ("After: estimate ± 2 model standard deviations. Before: logged values. Not a profit guarantee or a simultaneous confidence interval for all hypotheses.", "Кейін: баға ± модельдің 2 стандартты ауытқуы. Дейін: журналдағы мәндер. Бұл пайда кепілдігі де, барлық гипотезаның бірлескен сенімділік аралығы да емес."),
    "Положительная оценка до пилота, отрицательное наблюдение: {count}.": ("Positive prior estimate, negative observation: {count}.", "Пилотқа дейінгі баға оң, бақыланған нәтиже теріс: {count}."),
    "Отклонённые гипотезы": ("Rejected hypotheses", "Қабылданбаған гипотезалар"),
    "Тариф": ("Tariff", "Тариф"),
    "Предложение": ("Offer", "Ұсыныс"),
    "Причина": ("Reason", "Себеп"),
    "Явных отказов в журнале этого прогона нет.": ("No explicit rejections in this run's log.", "Бұл есеп журналында айқын бас тартулар жоқ."),
    "Исходные гипотезы": ("Initial hypotheses", "Бастапқы гипотезалар"),
    "Финальные кампании": ("Final campaigns", "Соңғы науқандар"),
    "Кампании ещё не выбраны": ("No campaigns selected yet", "Науқандар әлі таңдалмады"),
    "План формируется по результатам пилотного обучения": ("The plan follows the pilot learning results", "Жоспар пилоттық үйрену нәтижелері бойынша құрылады"),
    "Агент не сформировал таблицу финального плана.": ("The agent did not produce a final plan table.", "Агент соңғы жоспар кестесін құрмады."),
    "Кампания": ("Campaign", "Науқан"),
    "Сегмент": ("Segment", "Сегмент"),
    "Ожидаемый net": ("Expected net", "Күтілетін таза нәтиже"),
    "Нижняя оценка": ("Lower estimate", "Төменгі баға"),
    "Обоснование": ("Rationale", "Негіздеме"),
    "Ожидаемый net и нижняя оценка взяты из модели агента. Фактический результат скорера показан в разделе «Результат».": ("Expected net and lower estimate come from the agent model. Actual scorer output is in Results.", "Күтілетін таза нәтиже мен төменгі баға агент моделінен алынған. Бағалау жүйесінің нақты нәтижесі «Нәтиже» бөлімінде."),
    "Портрет сегмента": ("Audience profile", "Сегмент сипаттамасы"),
    "В сохранённом расчёте нет портрета сегмента. Требуется новый расчёт.": ("The saved run has no audience profile. Run a new evaluation.", "Сақталған есепте сегмент сипаттамасы жоқ. Қайта есептеу қажет."),
    "Портрет недоступен: {error}": ("Audience profile unavailable: {error}", "Сегмент сипаттамасы қолжетімсіз: {error}"),
    "Абонентов по фильтру": ("Matching subscribers", "Сүзгіге сай абоненттер"),
    "Средний текущий ARPU": ("Mean current ARPU", "Орташа ағымдағы ARPU"),
    "Средний прогнозный ARPU": ("Mean predicted ARPU", "Орташа болжамды ARPU"),
    "Не используют": ("No usage", "Қолданбайды"),
    "Небольшое": ("Light", "Аз"),
    "Активное": ("Heavy", "Белсенді"),
    "Низкое": ("Low", "Төмен"),
    "Среднее": ("Medium", "Орташа"),
    "Высокое": ("High", "Жоғары"),
    "Неизвестно": ("Unknown", "Белгісіз"),
    "Потребление данных": ("Data usage", "Интернет қолдану"),
    "Потребление звонков": ("Call usage", "Қоңырау қолдану"),
    "% абонентов": ("% of subscribers", "абоненттер үлесі, %"),
    "Данные": ("Data", "Интернет"),
    "Звонки": ("Calls", "Қоңыраулар"),
    "Доля, %": ("Share, %", "Үлес, %"),
    "Текущий ARPU, среднее": ("Current ARPU, mean", "Ағымдағы ARPU, орташа"),
    "Прогнозный ARPU, среднее": ("Predicted ARPU, mean", "Болжамды ARPU, орташа"),
    "Состав аудитории по фильтрам кампании до ограничения охвата и дедупликации. Data/call-сегменты здесь описательные: портрет не изменяет отбор агента.": ("Audience matching campaign filters before reach limits and deduplication. Data/call segments are descriptive; the profile does not change the agent's selection.", "Қамту шектеуі мен қайталануды жоюға дейін науқан сүзгілеріне сай аудитория құрамы. Интернет пен қоңырау сегменттері сипаттама үшін берілген, агенттің таңдауын өзгертпейді."),
    "Десять реализаций шума": ("Ten noise realizations", "Шудың он нұсқасы"),
    "Проверить seed 0–9": ("Test seeds 0–9", "Seed 0–9 тексеру"),
    "Сравниваем Flora и шаблон на одинаковых seed…": ("Comparing Flora and the baseline on the same seeds…", "Flora мен үлгі бірдей seed мәндерінде салыстырылуда…"),
    "Таблица устойчивости относится к предыдущей версии кода или данных.": ("Robustness results use a previous code or data version.", "Тұрақтылық нәтижелері кодтың немесе деректердің алдыңғы нұсқасына жатады."),
    "Часть прогонов завершилась с ошибкой. См. столбец «Ошибка».": ("Some runs failed. See the Error column.", "Кейбір есептер қатемен аяқталды. «Қате» бағанын қараңыз."),
    "Медиана": ("Median", "Медиана"),
    "Минимум": ("Minimum", "Ең аз"),
    "В плюсе": ("Positive runs", "Оң нәтижелер"),
    "С результатом": ("With results", "Нәтижесі бар"),
    "Серия ещё не запущена": ("The series has not run yet", "Есептер сериясы әлі басталмады"),
    "Seed 0–9 · Flora и шаблон организаторов · одинаковые условия": ("Seeds 0–9 · Flora and the organizer baseline · equal conditions", "Seed 0–9 · Flora және ұйымдастырушылар үлгісі · бірдей жағдайлар"),
    "Стресс-сценарии / неизвестная аудитория": ("Stress scenarios / unseen audience", "Стресс-сценарийлер / белгісіз аудитория"),
    "Стресс-отчёт RESULTS.md пока пуст.": ("The RESULTS.md stress report is empty.", "RESULTS.md стресс-есебі әзірге бос."),
    "Команда ещё не опубликовала RESULTS.md. Стресс-результаты не подставлены.": ("The team has not published RESULTS.md. No stress results substituted.", "Команда RESULTS.md файлын әлі жарияламады. Стресс-нәтижелер ойдан қосылмады."),
    "Стресс-отчёт недоступен: {error}": ("Stress report unavailable: {error}", "Стресс-есеп қолжетімсіз: {error}"),
    "Опубликованные результаты команды · обновлены {time}. Оракул знает эффекты заранее; агент получает только пилотные наблюдения.": ("Published team results · updated {time}. The oracle knows effects in advance; the agent only sees pilot observations.", "Команданың жарияланған нәтижелері · жаңартылған уақыты: {time}. Оракул әсерлерді алдын ала біледі, агент тек пилоттық бақылауларды алады."),
    "Оригинальный отчёт · русский": ("Original report · Russian", "Түпнұсқа есеп · орысша"),
    "Оригинальный журнал · русский": ("Original log · Russian", "Түпнұсқа журнал · орысша"),
    "Почему выбраны эти кампании": ("Why these campaigns", "Бұл науқандар неге таңдалды"),
    "Агент не вернул объяснение.": ("The agent returned no explanation.", "Агент түсіндірме бермеді."),
    "Объяснение ещё не сформировано": ("No explanation yet", "Түсіндірме әлі дайын емес"),
    "Локальное обоснование из журнала решений агента": ("Local rationale from the agent's decision log", "Агент шешімдері журналынан жергілікті негіздеме"),
    "Спросить про план": ("Ask about the plan", "Жоспар туралы сұрау"),
    "Во внешний API уйдут вопрос, агрегированный план и журнал. Строки профиля не отправляются. Ответ не меняет кампании и может содержать ошибки.": ("The external API receives the question, aggregated plan and log. No subscriber profile rows are sent. Answers do not change campaigns and may contain errors.", "Сыртқы API-ға сұрақ, жинақталған жоспар және журнал жіберіледі. Абонент профилінің жолдары жіберілмейді. Жауап науқандарды өзгертпейді және қате болуы мүмкін."),
    "Локальный режим · без внешних запросов": ("Local mode · no external requests", "Жергілікті режим · сыртқы сұраусыз"),
    "Вопрос": ("Question", "Сұрақ"),
    "Спросить": ("Ask", "Сұрау"),
    "Вопрос пуст.": ("The question is empty.", "Сұрақ бос."),
    "Готовим ответ…": ("Preparing an answer…", "Жауап дайындалуда…"),
    "Локальное объяснение агента": ("Local agent explanation", "Агенттің жергілікті түсіндірмесі"),
    "В этом прогоне агент не сохранил текстовое объяснение.": ("The agent saved no explanation for this run.", "Агент бұл есепке түсіндірме сақтамады."),
    "Ключ не подключён. Показано локальное объяснение без LLM.": ("No API key configured. Showing the local explanation without an LLM.", "API кілті қосылмаған. LLM-сыз жергілікті түсіндірме көрсетілді."),
    "LLM недоступна (HTTP {code}). Показано локальное объяснение.": ("LLM unavailable (HTTP {code}). Showing the local explanation.", "LLM қолжетімсіз (HTTP {code}). Жергілікті түсіндірме көрсетілді."),
    "Ответ LLM недоступен или некорректен. Показано локальное объяснение.": ("LLM response unavailable or invalid. Showing the local explanation.", "LLM жауабы қолжетімсіз немесе қате. Жергілікті түсіндірме көрсетілді."),
    "Диагностика: {name}": ("Diagnostics: {name}", "Диагностика: {name}"),
    "{name}: финальный план пуст; результат включает только пилоты.": ("{name}: the final plan is empty; results include pilots only.", "{name}: соңғы жоспар бос; нәтижеге тек пилоттар кіреді."),
    "{name}: скорер отбросил некорректные кампании.": ("{name}: the scorer discarded invalid campaigns.", "{name}: бағалау жүйесі жарамсыз науқандарды алып тастады."),
    "Объяснение по сохранённому журналу: {pilots} пилотов, {cells} ячеек, {campaigns} кампаний.": ("Saved log: {pilots} pilots across {cells} cells; {campaigns} campaigns selected.", "Сақталған журнал: {cells} ұяшықта {pilots} пилот; {campaigns} науқан таңдалды."),
    "Knowledge Gradient оценивает пользу следующего пилота для итогового плана. Пилоты калибруют исторические оценки под аудиторию. План учитывает оценку эффекта, неопределённость, стоимость контакта и лимиты ресурсов. Бесплатный push на остаток контактов может выбираться по положительной средней оценке. Прибыль не гарантирована.": ("Knowledge Gradient estimates how much the next pilot can improve the final plan. Pilots calibrate historical estimates to the audience. The plan accounts for estimated effects, uncertainty, contact costs and resource limits. Free push for remaining contacts may use a positive mean estimate. Profit is not guaranteed.", "Knowledge Gradient келесі пилоттың соңғы жоспарға пайдасын бағалайды. Пилоттар тарихи бағаларды аудиторияға бейімдейді. Жоспар әсер бағасын, белгісіздікті, байланыс шығынын және ресурс шектеулерін ескереді. Қалған байланыстарға тегін push оң орташа баға бойынша таңдалуы мүмкін. Пайдаға кепілдік жоқ."),
}

PHRASES = {
    "нижняя граница не окупает контакт или в ячейке есть связка лучше.": ("the lower estimate does not cover the contact cost, or the cell has a better option.", "төменгі баға байланыс шығынын өтемейді немесе ұяшықта тиімдірек нұсқа бар."),
    "оценка": ("estimate", "баға"),
    "Поиск по Knowledge Gradient исчерпан: оставшиеся пилоты — на проверку крупнейших непроверенных ставок плана.": ("Knowledge Gradient search exhausted: remaining pilots verify the largest untested plan choices.", "Knowledge Gradient іздеуі аяқталды: қалған пилоттар жоспардың тексерілмеген ірі таңдауларын тексереді."),
    "Разведка остановлена: ни один пилот не повышает ожидаемую ценность плана.": ("Exploration stopped: no pilot improves the expected plan value.", "Барлау тоқтатылды: ешбір пилот жоспардың күтілетін құндылығын арттырмайды."),
    "Все крупные ставки плана проверены — разведка завершена.": ("All major plan choices verified; exploration complete.", "Жоспардың барлық ірі таңдаулары тексерілді; барлау аяқталды."),
    "по калибровке пилотов": ("from pilot calibration", "пилоттармен калибрлеу бойынша"),
    "ожидаемая польза для плана": ("expected plan value", "жоспардың күтілетін пайдасы"),
    "проверка ставки плана, оценённой только через калибровку": ("verifying a plan choice estimated only through calibration", "тек калибрлеумен бағаланған жоспар таңдауын тексеру"),
    "Разброс «история → аудитория»": ("History-to-audience variability", "«Тарих → аудитория» ауытқуы"),
    "до пилота": ("before pilot", "пилотқа дейін"),
    "наблюдали": ("observed", "бақыланған"),
    "после": ("after", "кейін"),
    "эффект": ("effect", "әсер"),
    "пилоты": ("pilots", "пилоттар"),
    "Пилот": ("Pilot", "Пилот"),
    "абон.": ("subscribers", "абонент"),
    "Не берём": ("Rejected", "Қабылданбады"),
    "Резервная кампания: выгодных связок не найдено, прибыль не гарантирована.": ("Fallback campaign: no profitable combinations found; profit is not guaranteed.", "Қосалқы науқан: тиімді үйлесімдер табылмады, пайдаға кепілдік жоқ."),
}


TEXT.update({
    "Flora < 0": ("Flora < 0", "Flora < 0"),
    "минус": ("negative", "минус"),
    "В минусе": ("Negative", "Минуста"),
    "Flora в минусе в {n} из {m} прогонов — строки отмечены в таблице.": (
        "Flora is negative in {n} of {m} runs — rows are marked in the table.",
        "Flora {m} іске қосудың {n}-інде минуста — жолдар кестеде белгіленген."),
    "Flora в минусе: 0 из {m} прогонов этой серии.": (
        "Flora negative: 0 of {m} runs in this series.", "Flora минуста: осы сериядағы {m} іске қосудың 0-і."),
    "Разные стенды не смешиваются: у каждого свой набор сценариев, версия и размер проверки.": (
        "Test benches are not mixed: each has its own scenarios, version and sample size.",
        "Стендтер араластырылмайды: әрқайсысының өз сценарийлері, нұсқасы және тексеру көлемі бар."),
    "Худшие результаты по стендам": ("Worst results by test bench", "Стендтер бойынша ең нашар нәтижелер"),
    "Стресс-стенд v5 · 15 сценариев × 5 seed": ("Stress bench v5 · 15 scenarios × 5 seeds",
                                                  "v5 стресс-стенді · 15 сценарий × 5 seed"),
    "Реалистичный стенд · миры, слабо связанные с историей": (
        "Realistic bench · worlds weakly related to history", "Шынайы стенд · тарихпен әлсіз байланысқан әлемдер"),
    "оригинал на русском": ("original in Russian", "түпнұсқа орыс тілінде"),
    'Свернуть чат': ('Minimise chat', 'Чатты жию'),
    'Спросить AI': ('Ask AI', 'AI-дан сұрау'),
    'Вопросы о плане кампаний, пилотах и решениях агента': ('Questions about the campaign plan, pilots and agent decisions', 'Науқандар жоспары, пилоттар және агент шешімдері туралы сұрақтар'),
    'Локально · AI-ассистент через OpenAI': ('Local · AI assistant via OpenAI', 'Жергілікті · OpenAI арқылы AI-көмекші'),
    'Режим: OpenAI с проверкой ответа по данным плана': ('Mode: OpenAI with answers checked against plan data', 'Режим: жауабы жоспар деректерімен тексерілетін OpenAI'),
    'Сначала нажмите «Рассчитать кампании» — ассистент отвечает по данным рассчитанного плана.': ('First press «Calculate campaigns» — the assistant answers from the calculated plan.', 'Алдымен «Науқандарды есептеу» басыңыз — көмекші есептелген жоспар деректері бойынша жауап береді.'),
    'Например: почему отклонена гипотеза, зачем этот канал?': ('For example: why was a hypothesis rejected, why this channel?', 'Мысалы: гипотеза неге қабылданбады, неге бұл арна?'),
    'Ассистент отвечает только по плану кейса 04: пилоты, кампании, каналы, ограничения.': ('The assistant answers only about the case 04 plan: pilots, campaigns, channels, limits.', 'Көмекші тек 04 кейсінің жоспары бойынша жауап береді: пилоттар, науқандар, арналар, шектеулер.'),
    'Очистить диалог': ('Clear chat', 'Диалогты тазалау'),
    'Агент оптимизирует чистый прирост ARPU оператора минус стоимость контактов. Выгода, удовлетворённость и отток абонентов в модели и данных кейса не учитываются, поэтому по этому плану нельзя утверждать, выгоден ли переход самому абоненту. Агент не включает в основной план связки, у которых нижняя граница оценки эффекта не окупает контакт; это снижает риск падения выручки (downsell), но не исключает его: пилоты шумные, а часть связок оценена только через калибровку. Учёт выгоды абонента — направление развития (см. README).': ("The agent optimises the operator's net ARPU gain minus contact costs. Subscriber benefit, satisfaction and churn are not part of the model or the case data, so this plan cannot tell whether a switch benefits the subscriber. The main plan excludes links whose lower effect estimate does not pay for the contact; this reduces the risk of revenue loss (downsell) but does not rule it out: pilots are noisy and some links are estimated only through calibration. Accounting for subscriber benefit is a development direction (see README).", 'Агент оператордың ARPU таза өсімін байланыс шығынын шегергенде оңтайландырады. Абоненттің пайдасы, қанағаттануы және кетуі модельде және кейс деректерінде ескерілмейді, сондықтан бұл жоспар бойынша ауысудың абонентке тиімді екенін айтуға болмайды. Негізгі жоспарға әсер бағасының төменгі шегі байланысты өтемейтін байланыстар кірмейді; бұл табыстың төмендеу (downsell) қаупін азайтады, бірақ толық жоймайды: пилоттар шулы, ал кейбір байланыстар тек калибровка арқылы бағаланған. Абонент пайдасын ескеру — даму бағыты (README қараңыз).'),
    "Факт о модели агента": ("Fact about the agent model", "Агент моделі туралы факт"),
    "Ассистент отвечает только на вопросы о плане кампаний, пилотах, решениях агента и ограничениях кейса 04.": (
        "The assistant only answers questions about the campaign plan, pilots, agent decisions and limits of case 04.",
        "Көмекші тек науқандар жоспары, пилоттар, агент шешімдері және 04 кейсінің шектеулері туралы сұрақтарға жауап береді."),
    "Ответ LLM отклонён: в нём есть числа, которых нет в данных плана ({numbers}). Показано локальное объяснение.": (
        "LLM answer rejected: it contains numbers not present in the plan data ({numbers}). Local explanation shown.",
        "LLM жауабы қабылданбады: онда жоспар деректерінде жоқ сандар бар ({numbers}). Жергілікті түсіндірме көрсетілді."),
    "В данных плана нет ответа на этот вопрос.": ("The plan data does not contain an answer to this question.",
                                                  "Жоспар деректерінде бұл сұраққа жауап жоқ."),
    "Локальный режим не анализирует вопрос: показано общее объяснение плана. Для ответа на вопрос нужен OPENAI_API_KEY и включённый переключатель «LLM · OpenAI».": (
        "Local mode does not analyse the question: the general plan explanation is shown. "
        "To answer the question, set OPENAI_API_KEY and turn on «LLM · OpenAI».",
        "Жергілікті режим сұрақты талдамайды: жоспардың жалпы түсіндірмесі көрсетілді. "
        "Сұраққа жауап алу үшін OPENAI_API_KEY және «LLM · OpenAI» қосқышы керек."),
})


def language():
    return st.session_state.get("language", "ru")


def t(text, **kwargs):
    translated = TEXT.get(text)
    result = translated[0 if language() == "en" else 1] if translated and language() != "ru" else text
    return result.format(**kwargs) if kwargs else result


def agent_text(text):
    if language() == "ru":
        return str(text)
    result = str(text)
    for source in sorted(PHRASES, key=len, reverse=True):
        result = result.replace(source, PHRASES[source][0 if language() == "en" else 1])
    return result


def explanation_text(run):
    original = run.get("explanation", "")
    if language() == "ru" or not original:
        return original
    match = re.search(r"Агент провёл (\d+) пилотов в (\d+) ячейках.*?отобрал (\d+) кампаний", original, re.S)
    if not match:
        return t("Оригинальный журнал · русский") + "\n\n" + original
    return t("Объяснение по сохранённому журналу: {pilots} пилотов, {cells} ячеек, {campaigns} кампаний.",
             pilots=match[1], cells=match[2], campaigns=match[3]) + "\n\n" + t(
        "Knowledge Gradient оценивает пользу следующего пилота для итогового плана. Пилоты калибруют исторические оценки под аудиторию. План учитывает оценку эффекта, неопределённость, стоимость контакта и лимиты ресурсов. Бесплатный push на остаток контактов может выбираться по положительной средней оценке. Прибыль не гарантирована.")
