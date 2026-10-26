# master_runner.py
"""wbot v2 "Weltwetter" -- taeglicher Lauf (Cron stuendlich, mit flock; laeuft nur einmal pro UTC-Tag erfolgreich).

1. Alle Wetterstationen aktualisieren, Vorhersage fuer heute (live/forecast.py, gleiche Funktionen wie Backtest)
2. Papierkonto abrechnen + neue Papier-Trades (immer)
3. Echtgeld nur bei mode == "live" UND (Test bestanden ODER live_trotz_nicht_bestanden == true)
4. Wetterbericht + Vorwaertspruefung per Telegram
"""
import os
import sys
import json
import logging
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
DONE_FILE = os.path.join(PROJECT_ROOT, 'artifacts', 'live', 'last_run.txt')


def load_json(name):
    with open(os.path.join(PROJECT_ROOT, name), encoding='utf-8') as f:
        return json.load(f)


def main():
    os.makedirs(os.path.join(PROJECT_ROOT, 'logs'), exist_ok=True)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        handlers=[logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'wbot.log'), encoding='utf-8'),
                                  logging.StreamHandler()])
    from wbot.world import sources as src
    from wbot.live.forecast import forecast_today, forward_verification, research_result, today_utc
    from wbot.live import paper, trader
    from wbot.live.report import weather_report
    from wbot.utils.telegram import send_message

    today = today_utc()
    force = '--force' in sys.argv
    if not force and os.path.exists(DONE_FILE) and open(DONE_FILE).read().strip() == str(today.date()):
        return
    live = load_json('settings.json')['live_trading_settings']
    secrets = load_json('secret.json') if os.path.exists(os.path.join(PROJECT_ROOT, 'secret.json')) else {}
    tg = secrets.get('telegram', {})

    def notify(text):
        logging.info('Bericht:\n' + text)
        send_message(tg.get('bot_token'), tg.get('chat_id'), text)

    try:
        panel, table = forecast_today(live, today)
        decisions = {r.coin: (r.side, r.atr) for r in table.itertuples()}

        st = paper.load_state(live.get('paper_start_capital', 1000))
        coins = set(table['coin']) | set(st['open'])
        closed, opened = paper.step(st, today, decisions,
                                    {c: src.binance_klines(c) for c in coins},
                                    {c: src.binance_funding(c)['rate'] for c in coins},
                                    live.get('risk_per_trade_pct', 1.0))
        paper.save_state(st)

        live_msgs = []
        mode = live.get('mode', 'paper')
        if mode == 'live':
            passed = bool(research_result().get('bestanden'))
            if not passed and not live.get('live_trotz_nicht_bestanden', False):
                live_msgs.append('⛔ mode=live, aber der praeregistrierte Test ist NICHT bestanden -> kein Echtgeld. '
                                 '(Freigabe nur mit live_trotz_nicht_bestanden=true)')
            else:
                from wbot.utils.exchange import Exchange
                ex = Exchange(secrets['wbot'][0])
                lst = trader.load_state()
                live_msgs += trader.manage(ex, lst, today)
                live_msgs += trader.open_trades(ex, lst, today, decisions, live.get('risk_per_trade_pct', 1.0),
                                                int(live.get('leverage', 5)))
                trader.save_state(lst)

        fwd, n_fwd = forward_verification(panel)
        notify(weather_report(today, table, st, closed, opened, live_msgs, fwd, n_fwd, mode))
        os.makedirs(os.path.dirname(DONE_FILE), exist_ok=True)
        with open(DONE_FILE, 'w') as f:
            f.write(str(today.date()))
    except Exception as e:
        logging.critical(f'Tageslauf fehlgeschlagen: {e}', exc_info=True)
        # stuendliche Wiederholung: Fehler nur einmal pro Tag melden
        fail_file = DONE_FILE.replace('last_run', 'last_fail')
        if os.path.exists(fail_file) and open(fail_file).read().strip() == str(today.date()):
            return
        os.makedirs(os.path.dirname(fail_file), exist_ok=True)
        with open(fail_file, 'w') as f:
            f.write(str(today.date()))
        notify(f'🚨 wbot Tageslauf fehlgeschlagen ({today.date()}):\n{e}\n{traceback.format_exc()[-800:]}')


if __name__ == '__main__':
    main()
