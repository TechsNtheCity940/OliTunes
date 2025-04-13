% Custom metal notation definitions for LilyPond

% Palm mute symbol
palmMute = #(define-music-function (parser location note) (ly:music?)
   #{\tweak ParenthesesItem.font-size -2 \(( #note )\) #})

% Chugging notation
chug = #(define-event-class 'chug-event 'articulation)
#(define-music-function (parser location note) (ly:music?)
   #{\tweak stencil #ly:text-interface::print
     \tweak text "ch"
     #note #})

% Harmonics
harmonic = #(define-event-class 'harmonic-event 'articulation)
#(define-music-function (parser location note) (ly:music?)
   #{\tweak stencil #(lambda (grob)
                      (grob-interpret-markup grob
                       (markup #:fontsize -4 "harm.")))
     #note #})

% Bends
bend = #(define-event-class 'bend-event 'articulation)
#(define-music-function (parser location note) (ly:music?)
   #{\tweak stencil #ly:text-interface::print
     \tweak text "b"
     #note #})
