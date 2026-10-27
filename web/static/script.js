(function () {
    "use strict";

    var REQUEST_TIMEOUT_MS = 1800000;
    var PIECE_THEME = "https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png";

    var EXAMPLES = [
        "1. e4 e5 2. Nf3 Nc6 3. Bb5",
        "1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3",
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3",
        "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4 e5d4 c3d4 c5b4",
        "[Event \"Casual game\"]\n[Result \"*\"]\n\n1. Nf3 d5 2. g3 c5 3. Bg2 Nc6 4. O-O e6 5. d3 Nf6 6. Nbd2 Be7 7. e4 O-O *"
    ];
    var exampleIdx = -1;

    var board = null;
    var askBoard = null;
    var askState = null;
    var game = new Chess();
    var moveHistory = [];
    var playerColor = "w";
    var thinking = false;
    var analyzing = false;
    var thinkTimer = null;
    var thinkStarted = 0;
    var serverDevice = null;
    var dragging = false;
    var lastHighlightedFrom = null;
    var lastHighlightedTo = null;

    $(document).ready(function () {
        initBoards();
        fetchServerInfo();
        updateStatus();
        renderHistory();

        $("#btn-new-game").on("click", newGame);
        $("#btn-undo").on("click", undoMove);
        $("#btn-flip").on("click", function () { board.flip(); });
        $("#btn-copy-pgn").on("click", copyPgn);

        $("#btn-advice").on("click", requestAdvice);
        $("#btn-fill-example").on("click", fillNextExample);
        $("#btn-continue-play").on("click", continueFromAsk);
        $("#analysis-moves").on("keydown", function (e) {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) requestAdvice();
        });

        $(".tab-btn").on("click", function () {
            switchTab($(this).data("tab"));
        });
    });

    function switchTab(name) {
        $(".tab-btn").removeClass("active")
            .filter('[data-tab="' + name + '"]').addClass("active");
        $(".tab-panel").removeClass("active");
        $("#tab-" + name).addClass("active");
        if (name === "play" && board) board.resize();
        if (name === "ask" && askBoard) askBoard.resize();
    }

    function initBoards() {
        board = Chessboard("board", {
            draggable: true,
            position: "start",
            orientation: "white",
            onDragStart: onDragStart,
            onDrop: onDrop,
            onSnapEnd: onSnapEnd,
            onMouseoverSquare: onMouseoverSquare,
            onMouseoutSquare: onMouseoutSquare,
            pieceTheme: PIECE_THEME
        });
        askBoard = Chessboard("ask-board", {
            draggable: false,
            position: "start",
            orientation: "white",
            pieceTheme: PIECE_THEME
        });
        $(window).on("resize", function () {
            board.resize();
            askBoard.resize();
        });
    }

    function fetchServerInfo() {
        $.get("/api/health").done(function (data) {
            serverDevice = data.device || null;
        });
    }

    function escapeHtml(s) {
        return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function num(x, digits) {
        return x.toFixed(digits).replace(".", ",");
    }

    function signed(x, digits) {
        return (x >= 0 ? "+" : "") + num(x, digits);
    }

    function requestErrorMessage(xhr, textStatus) {
        if (textStatus === "timeout") return "έληξε ο χρόνος (30 λεπτά)";
        return xhr.responseJSON ? xhr.responseJSON.error : "σφάλμα δικτύου";
    }

    function candidatesHtml(candidates, title, open) {
        var sorted = candidates.slice().sort(function (a, b) { return b.eval_cp - a.eval_cp; });
        var html = "<details" + (open ? " open" : "") + "><summary>" + title + "</summary>" +
            '<ul class="candidates-list">';
        for (var i = 0; i < sorted.length; i++) {
            var c = sorted[i];
            html += '<li class="' + (c.selected ? "selected" : "") + '">' +
                (i + 1) + ". " + escapeHtml(c.san) + " (" + escapeHtml(c.uci) + ") → " +
                signed(c.eval_pawns, 2) + "</li>";
        }
        return html + "</ul></details>";
    }

    function overrodeNote(llmTopUci) {
        return "Το Stockfish προτίμησε άλλη κίνηση από την πρώτη επιλογή του γλωσσικού μοντέλου (" +
            escapeHtml(llmTopUci) + ")";
    }

    function showLegalMoves(square) {
        var moves = game.moves({ square: square, verbose: true });
        if (moves.length === 0) return;
        $("#board .square-" + square).addClass("source-square");
        for (var i = 0; i < moves.length; i++) {
            var isCapture = moves[i].flags.indexOf("c") !== -1 || moves[i].flags.indexOf("e") !== -1;
            $("#board .square-" + moves[i].to).addClass(isCapture ? "legal-capture" : "legal-dot");
        }
    }

    function clearLegalHints() {
        $("#board .square-55d63").removeClass("legal-dot legal-capture source-square");
    }

    function isOwnPiece(piece) {
        return piece && piece.charAt(0) === playerColor;
    }

    function onMouseoverSquare(square, piece) {
        if (dragging || thinking || game.game_over()) return;
        if (game.turn() !== playerColor || !isOwnPiece(piece)) return;
        clearLegalHints();
        showLegalMoves(square);
    }

    function onMouseoutSquare() {
        if (!dragging) clearLegalHints();
    }

    function onDragStart(source, piece) {
        if (game.game_over() || thinking) return false;
        if (game.turn() !== playerColor || !isOwnPiece(piece)) return false;
        dragging = true;
        clearLegalHints();
        showLegalMoves(source);
    }

    function onDrop(source, target) {
        var move = game.move({ from: source, to: target, promotion: "q" });
        if (move === null) return "snapback";

        clearHighlights();
        moveHistory.push(move.from + move.to + (move.promotion || ""));
        renderHistory();
        updateStatus();

        if (game.game_over()) {
            showGameOver();
        } else {
            setTimeout(requestModelMove, 250);
        }
    }

    function onSnapEnd() {
        dragging = false;
        clearLegalHints();
        board.position(game.fen());
    }

    function historyString() {
        return moveHistory.length === 0 ? "(none — opening)" : moveHistory.slice(-10).join(" ");
    }

    function requestModelMove() {
        setThinking(true);
        $.ajax({
            url: "/api/predict",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify({ fen: game.fen(), history: historyString() }),
            timeout: REQUEST_TIMEOUT_MS
        })
            .done(function (data) {
                if (!data || data.error) {
                    displayPlayError(data && data.error);
                } else {
                    applyModelMove(data);
                }
            })
            .fail(function (xhr, textStatus) {
                displayPlayError(requestErrorMessage(xhr, textStatus));
            })
            .always(function () {
                setThinking(false);
            });
    }

    function applyModelMove(data) {
        var uci = data.move_uci;
        if (!uci) return;
        var move = game.move({
            from: uci.substring(0, 2),
            to: uci.substring(2, 4),
            promotion: uci.length > 4 ? uci[4] : undefined
        });
        if (!move) {
            displayPlayError("ο server επέστρεψε παράνομη κίνηση " + uci);
            return;
        }

        moveHistory.push(uci);
        board.position(game.fen());
        highlightMove(uci.substring(0, 2), uci.substring(2, 4));
        displayMoveInfo(data);
        updateEvalBar(data.eval_cp);
        renderHistory();
        updateStatus();
        if (game.game_over()) showGameOver();
    }

    function displayPlayError(msg) {
        $("#last-move-info").html(
            '<span class="overrode-note">Σφάλμα: ' + escapeHtml(msg || "άγνωστο") +
            '<span class="retry-link" id="retry-model">ξαναπροσπάθησε</span></span>'
        );
        $("#retry-model").on("click", function () {
            if (!thinking && game.turn() !== playerColor && !game.game_over()) {
                requestModelMove();
            }
        });
    }

    function fillNextExample() {
        exampleIdx = (exampleIdx + 1) % EXAMPLES.length;
        $("#analysis-moves").val(EXAMPLES[exampleIdx]);
        $("#analysis-result").removeClass("error")
            .html('<span class="muted">Παράδειγμα ' + (exampleIdx + 1) + "/" + EXAMPLES.length +
                  " — πάτα «Πάρε κίνηση».</span>");
        $("#btn-continue-play").addClass("hidden");
    }

    function requestAdvice() {
        if (analyzing || thinking) return;
        var moves = $("#analysis-moves").val().trim();
        var resultEl = $("#analysis-result");
        if (!moves) {
            resultEl.text("Επικόλλησε πρώτα τις κινήσεις της παρτίδας.").addClass("error");
            return;
        }

        analyzing = true;
        askState = null;
        $("#btn-advice").prop("disabled", true);
        $("#btn-continue-play").addClass("hidden");
        var hint = serverDevice === "cpu" ? " (CPU — μπορεί να πάρει λεπτά)" : "";
        resultEl.removeClass("error")
            .html('<span class="advice-spinner"></span>Ξαναπαίζω τις κινήσεις και ρωτάω το μοντέλο…' + hint);

        $.ajax({
            url: "/api/advise",
            method: "POST",
            contentType: "application/json",
            data: JSON.stringify({ moves: moves }),
            timeout: REQUEST_TIMEOUT_MS
        })
            .done(function (data) {
                if (!data || data.error) {
                    resultEl.text("Σφάλμα: " + (data && data.error)).addClass("error");
                } else {
                    renderAdvice(data);
                }
            })
            .fail(function (xhr, textStatus) {
                resultEl.text("Σφάλμα: " + requestErrorMessage(xhr, textStatus)).addClass("error");
            })
            .always(function () {
                analyzing = false;
                $("#btn-advice").prop("disabled", false);
            });
    }

    function renderParsedMoves(sanMoves) {
        var out = "";
        for (var i = 0; sanMoves && i < sanMoves.length; i += 2) {
            out += (i / 2 + 1) + ". " + sanMoves[i] + (sanMoves[i + 1] ? " " + sanMoves[i + 1] : "") + "  ";
        }
        $("#ask-moves-parsed").text(out.trim());
    }

    function renderAdvice(data) {
        askState = { fen: data.fen, uci_moves: data.uci_moves || [], start_fen: data.start_fen || "" };

        askBoard.orientation(data.side_to_move === "White" ? "white" : "black");
        askBoard.position(data.fen, false);
        $("#ask-board .square-55d63").removeClass("highlight-from highlight-to");
        $("#ask-board .square-" + data.move_uci.substring(0, 2)).addClass("highlight-from");
        $("#ask-board .square-" + data.move_uci.substring(2, 4)).addClass("highlight-to");

        var side = data.side_to_move === "White" ? "τα Λευκά" : "τα Μαύρα";
        var moveNumber = Math.floor(data.ply / 2) + 1;
        $("#ask-position-info").removeClass("muted").text("Παίζουν " + side + " · κίνηση " + moveNumber);
        renderParsedMoves(data.san_moves);

        var html = '<span class="advice-move">' + escapeHtml(data.move_san) + "</span> " +
            "(" + escapeHtml(data.move_uci) + ") — παίζουν " + side +
            '<span class="advice-line">Αξιολόγηση Stockfish μετά την κίνηση: ' +
            signed(data.eval_pawns, 2) + "</span>";
        if (typeof data.inference_ms === "number") {
            html += '<span class="advice-line muted">σε ' + num(data.inference_ms / 1000, 1) +
                " s · ιστορικό που είδε το μοντέλο: " + escapeHtml(data.history_used) + "</span>";
        }
        if (!data.selected_was_llm_top) {
            html += '<span class="advice-line overrode-note">' + overrodeNote(data.llm_top_uci) + "</span>";
        }
        if (data.warning) {
            html += '<span class="advice-line advice-warning">' + escapeHtml(data.warning) + "</span>";
        }
        if (data.candidates && data.candidates.length) {
            html += candidatesHtml(data.candidates,
                "Οι " + data.candidates.length + " υποψήφιες κινήσεις (αξιολόγηση Stockfish)", false);
        }
        $("#analysis-result").removeClass("error").html(html);
        $("#btn-continue-play").removeClass("hidden");
    }

    function continueFromAsk() {
        if (!askState || thinking) return;
        var g = new Chess();
        if (askState.start_fen && !g.load(askState.start_fen)) return;
        for (var i = 0; i < askState.uci_moves.length; i++) {
            var u = askState.uci_moves[i];
            var move = g.move({ from: u.substring(0, 2), to: u.substring(2, 4),
                                promotion: u.length > 4 ? u[4] : undefined });
            if (!move) return;
        }

        game = g;
        moveHistory = askState.uci_moves.slice();
        playerColor = game.turn();
        $("#color-select").val(playerColor);
        board.orientation(playerColor === "w" ? "white" : "black");
        board.position(game.fen());
        clearHighlights();
        clearLegalHints();
        $("#game-over-banner").addClass("hidden");
        $("#last-move-info").html('<em class="muted">Συνεχίζεις από τη θέση της ανάλυσης — παίζεις ' +
            (playerColor === "w" ? "τα Λευκά" : "τα Μαύρα") + ". Κάνε την κίνησή σου.</em>");
        $("#copy-feedback").text("").removeClass("error");
        updateEvalBar(0);
        renderHistory();
        updateStatus();
        switchTab("play");
    }

    function setThinking(on) {
        thinking = on;
        $("#btn-new-game, #btn-undo").prop("disabled", on);
        if (on) {
            thinkStarted = Date.now();
            $("#overlay-text").text(thinkingLabel(0));
            $("#board-overlay").removeClass("hidden");
            thinkTimer = setInterval(function () {
                $("#overlay-text").text(thinkingLabel(Math.round((Date.now() - thinkStarted) / 1000)));
            }, 1000);
        } else {
            $("#board-overlay").addClass("hidden");
            clearInterval(thinkTimer);
            thinkTimer = null;
        }
    }

    function thinkingLabel(secs) {
        var mins = Math.floor(secs / 60);
        var time = mins > 0 ? mins + "λ " + (secs % 60) + "s" : secs + "s";
        return "Το μοντέλο σκέφτεται… " + time + (serverDevice === "cpu" ? " (CPU — υπομονή)" : "");
    }

    function colorName(c) {
        return c === "w" ? "Λευκά" : "Μαύρα";
    }

    function updateStatus() {
        var who = game.turn() === playerColor ? "Παίζεις εσύ" : "Παίζει το μοντέλο";
        var status;
        if (game.in_checkmate()) {
            status = "Ματ — κερδίζουν τα " + colorName(game.turn() === "w" ? "b" : "w");
        } else if (game.in_stalemate()) {
            status = "Πατ — ισοπαλία";
        } else if (game.in_draw()) {
            status = "Ισοπαλία";
        } else if (game.in_check()) {
            status = who + " (" + colorName(game.turn()) + ", σαχ)";
        } else {
            status = who + " (" + colorName(game.turn()) + ")";
        }
        $("#status").text(status);
    }

    function renderHistory() {
        var sanMoves = game.history();
        var container = $("#move-history");
        if (sanMoves.length === 0) {
            container.html('<em class="muted">Καμία κίνηση ακόμη</em>');
            return;
        }
        var html = "";
        for (var i = 0; i < sanMoves.length; i += 2) {
            var whiteLast = i === sanMoves.length - 1 ? " last" : "";
            var blackLast = i + 1 === sanMoves.length - 1 ? " last" : "";
            html += '<div class="move-row">' +
                '<span class="move-num">' + (i / 2 + 1) + ".</span>" +
                '<span class="move-san' + whiteLast + '">' + sanMoves[i] + "</span>" +
                '<span class="move-san' + blackLast + '">' + (sanMoves[i + 1] || "") + "</span>" +
                "</div>";
        }
        container.html(html);
        container.scrollTop(container[0].scrollHeight);
    }

    function updateEvalBar(evalCp) {
        var modelColor = playerColor === "w" ? "b" : "w";
        var whiteCp = modelColor === "w" ? evalCp : -evalCp;

        var pct = 50 + 50 * (2 / (1 + Math.exp(-whiteCp / 400)) - 1);
        pct = Math.max(4, Math.min(96, pct));
        if (whiteCp >= 9000) pct = 100;
        if (whiteCp <= -9000) pct = 0;
        $("#eval-bar-fill").css("height", pct + "%");

        var label;
        if (Math.abs(whiteCp) >= 9000) {
            label = whiteCp > 0 ? "M" : "-M";
        } else {
            label = (whiteCp > 0 ? "+" : "") + num(whiteCp / 100.0, 1);
        }
        $("#eval-bar-label").text(label);
    }

    function displayMoveInfo(data) {
        var evalClass = data.eval_pawns > 0.3 ? "eval-positive"
            : data.eval_pawns < -0.3 ? "eval-negative" : "eval-neutral";
        var html = "Έπαιξε <strong>" + escapeHtml(data.move_san) + "</strong> (" + escapeHtml(data.move_uci) + ") " +
            '— αξιολόγηση <span class="' + evalClass + '">' + signed(data.eval_pawns, 2) + "</span>";
        if (typeof data.inference_ms === "number") {
            html += '<span class="inference-time">απάντησε σε ' + num(data.inference_ms / 1000, 1) + " s</span>";
        }
        if (!data.selected_was_llm_top) {
            html += '<span class="overrode-note">' + overrodeNote(data.llm_top_uci) + "</span>";
        }
        if ($("#show-candidates").is(":checked") && data.candidates) {
            html += candidatesHtml(data.candidates, "Οι " + data.candidates.length + " υποψήφιες κινήσεις", true);
        }
        $("#last-move-info").html(html);
    }

    function showGameOver() {
        var msg = "Ισοπαλία";
        if (game.in_checkmate()) {
            var winner = game.turn() === "w" ? "b" : "w";
            msg = winner === playerColor ? "Ματ — κέρδισες!" : "Ματ — κέρδισε το μοντέλο";
        } else if (game.in_stalemate()) {
            msg = "Πατ — ισοπαλία";
        }
        $("#game-over-banner").text(msg).removeClass("hidden");
    }

    function highlightMove(from, to) {
        clearHighlights();
        $("#board .square-" + from).addClass("highlight-from");
        $("#board .square-" + to).addClass("highlight-to");
        lastHighlightedFrom = from;
        lastHighlightedTo = to;
    }

    function clearHighlights() {
        if (lastHighlightedFrom) $("#board .square-" + lastHighlightedFrom).removeClass("highlight-from");
        if (lastHighlightedTo) $("#board .square-" + lastHighlightedTo).removeClass("highlight-to");
        lastHighlightedFrom = null;
        lastHighlightedTo = null;
    }

    function newGame() {
        if (thinking) return;
        playerColor = $("#color-select").val();
        game = new Chess();
        moveHistory = [];
        board.orientation(playerColor === "w" ? "white" : "black");
        board.start();
        clearHighlights();
        clearLegalHints();
        $("#game-over-banner").addClass("hidden");
        $("#last-move-info").html('<em class="muted">Κάνε μια κίνηση για να ξεκινήσεις.</em>');
        $("#copy-feedback").text("").removeClass("error");
        updateEvalBar(0);
        renderHistory();
        updateStatus();
        if (playerColor === "b") setTimeout(requestModelMove, 400);
    }

    function undoMove() {
        if (thinking || moveHistory.length < 2 || game.turn() !== playerColor) return;
        game.undo();
        game.undo();
        moveHistory.pop();
        moveHistory.pop();
        board.position(game.fen());
        clearHighlights();
        clearLegalHints();
        $("#game-over-banner").addClass("hidden");
        $("#last-move-info").html('<em class="muted">Πήγες μία κίνηση πίσω. Παίξε ξανά.</em>');
        renderHistory();
        updateStatus();
    }

    function copyPgn() {
        var feedback = $("#copy-feedback");
        if (!game.history().length) {
            feedback.text("Δεν υπάρχουν κινήσεις ακόμη").addClass("error");
            return;
        }
        var text = game.pgn();
        var done = function () { feedback.text("Αντιγράφηκε").removeClass("error"); };
        var failed = function () { feedback.text("Η αντιγραφή απέτυχε").addClass("error"); };

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(done, failed);
            return;
        }
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        try {
            document.execCommand("copy");
            done();
        } catch (e) {
            failed();
        }
        document.body.removeChild(ta);
    }
})();
