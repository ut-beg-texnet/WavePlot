{RESOURCE: Display Primitives - Graphics Initialization and Windowing commands and Data Structures}

 UNIT Dsprim;

  {Contains data structures & operators, needed to display the different
     graph types offered. This covers routines provided in the Graph Unit,
     procedures to perform functions (e.g. draw arrowheads), not provided
     in the unit, & windowing data structures & operators}

  {* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * }


  INTERFACE

     USES
       Graph, Crt, Setts;

     TYPE
       HV             = 1..2;
       Pixels         = ARRAY [HV] OF Integer;

     CONST
       ScriFont=5;  SimpFont=6;  TscrFont=7;
       LcomFont=8;  EuroFont=9;  BoldFont=10;

     CONST
       ScrnLen = 24.5;

{       MenuBG       = Blue;
       MenuFG       = Yellow;
       MenuHL       = LightBlue;}
       MenuBG       = Blue;
       MenuFG       = Black;
       MenuHL       = Cyan;
       MsgHL        = LightCyan;
       MsgBG        = LightBlue;
       MsgFG        = White;
       Lastcolour   = 15;

       DsCol_Bgrnd  = Black;
       DsCol_Fgrnd  = White;
       DsCol_Fgrnd2 = Yellow;   {Normal Background}
{       DsCol_Bgrnd  = White;
       DsCol_Fgrnd  = Blue;
       DsCol_Fgrnd2 = Blue;      {Special Background for Screen Capture}

       DsCol_Axes   = DsCol_Fgrnd;
       DsCol_Val    = DsCol_Fgrnd;
       DsCol_Scal   = DsCol_Fgrnd;
       DsCol_List   = DsCol_Fgrnd;
       DsCol_Seism  = LightCyan;

       DsCol_Border  = Green;
       DsCol_CStress = Cyan;
       DsCol_TStress = LightRed;
       DsCol_Vect    = DsCol_Fgrnd;
       DsCol_Arrow   = LightRed;
       DsCol_Geom    = DsCol_Fgrnd;
       DsCol_Titles  = DsCol_Fgrnd;

     CONST
        SVGA256       = HercMono;
        SVGA16        = Att400;
        VGAoffset     = 64;
        EGAoffset     = 1;
        SVGAoffset    = 128;

     TYPE
       RGBrec = RECORD
                  Rd, Gr, Bl : Byte;
                END;
       RGBpalette = RECORD
                      Sz, Offset : Byte;
                      Clr        : ARRAY [0..255] OF RGBrec;
                    END;
       RGBpalPtr  = ^RGBpalette;

     CONST
       EGAdefault : PaletteType = ( Size:   16;
                                    Colors: ( $00, $01, $02, $03, $04, $05, $14, $07,
                                              $38, $39, $3A, $3B, $3C, $3D, $3E, $3F  )  );
       EGARainBow : PaletteType = ( Size:   16;
                                    Colors: ( $00, $01, $09, $0B, $13, $12, $16, $26,
                                              $24, $3C, $35, $05, $2D, $04, $3E, $3F  )  );



       VGAdefault :
              RECORD
                Sz, Offset : Byte;
                Clr        : ARRAY [0..15] OF RGBrec;
              END
                   = (  Sz:16;  Offset:0;
                        Clr : (
                                  (Rd:$00; Gr:$00; Bl:$00),  {Black}
                                  (Rd:$00; Gr:$00; Bl:$28),  {Blue}
                                  (Rd:$00; Gr:$28; Bl:$00),  {Green}
                                  (Rd:$00; Gr:$28; Bl:$28),  {Cyan}
                                  (Rd:$28; Gr:$07; Bl:$07),  {Red}
                                  (Rd:$28; Gr:$00; Bl:$28),  {Magenta}
                                  (Rd:$28; Gr:$1E; Bl:$00),  {Brown}
                                  (Rd:$31; Gr:$31; Bl:$07),  {Lt Grey}
                                  (Rd:$1A; Gr:$1A; Bl:$1A),  {Dark Grey}
                                  (Rd:$00; Gr:$00; Bl:$3F),  {Lt. Blue}
                                  (Rd:$09; Gr:$3F; Bl:$09),  {Lt. Green}
                                  (Rd:$00; Gr:$3F; Bl:$3F),  {Lt. Cyan}
                                  (Rd:$3F; Gr:$0A; Bl:$0A),  {Lt. Red}
                                  (Rd:$2C; Gr:$00; Bl:$3F),  {Lt. Magenta}
                                  (Rd:$3F; Gr:$3F; Bl:$12),  {Yellow}
                                  (Rd:$3F; Gr:$3F; Bl:$3F)   {Br White}     )  );

(*       VGAdefault :
              RECORD
                Sz, Offset : Byte;
                Clr        : ARRAY [0..15] OF RGBrec;
              END
                   = (  Sz:16;  Offset:0;
                        Clr : (
                                  (Rd:$00; Gr:$00; Bl:$00),  {Black}
                                  (Rd:$00; Gr:$00; Bl:$2C),  {Blue}
                                  (Rd:$00; Gr:$2C; Bl:$00),  {Green}
                                  (Rd:$00; Gr:$2C; Bl:$2C),  {Cyan}
                                  (Rd:$2C; Gr:$00; Bl:$00),  {Red}
                                  (Rd:$2C; Gr:$00; Bl:$2C),  {Magenta}
                                  (Rd:$2C; Gr:$1C; Bl:$00),  {Brown}
                                  (Rd:$2C; Gr:$2C; Bl:$2C),  {Lt Grey}
                                  (Rd:$1C; Gr:$1C; Bl:$1C),  {Dark Grey}
                                  (Rd:$00; Gr:$00; Bl:$3C),  {Lt. Blue}
                                  (Rd:$00; Gr:$3C; Bl:$00),  {Lt. Green}
                                  (Rd:$00; Gr:$3C; Bl:$3C),  {Lt. Cyan}
                                  (Rd:$3C; Gr:$00; Bl:$00),  {Lt. Red}
                                  (Rd:$3C; Gr:$00; Bl:$3C),  {Lt. Magenta}
                                  (Rd:$3C; Gr:$3C; Bl:$00),  {Yellow}
                                  (Rd:$3C; Gr:$3C; Bl:$3C)   {Br White}     )  );
*)

       VGAdefault_w :         {Normal, but white background}
              RECORD
                Sz, Offset : Byte;
                Clr        : ARRAY [0..15] OF RGBrec;
              END
                   = (  Sz:16;  Offset:0;
                        Clr : (
                                  (Rd:$3F; Gr:$3F; Bl:$3F),  {Br White}
                                  (Rd:$00; Gr:$00; Bl:$3F),  {Lt. Blue}
                                  (Rd:$09; Gr:$3F; Bl:$09),  {Lt. Green}
                                  (Rd:$00; Gr:$3F; Bl:$3F),  {Lt. Cyan}
                                  (Rd:$3F; Gr:$0A; Bl:$0A),  {Lt. Red}
                                  (Rd:$2C; Gr:$00; Bl:$3F),  {Lt. Magenta}
                                  (Rd:$3F; Gr:$3F; Bl:$12),  {Yellow}
                                  (Rd:$31; Gr:$31; Bl:$07),  {Lt Grey}
                                  (Rd:$28; Gr:$1E; Bl:$00),  {Brown}
                                  (Rd:$00; Gr:$00; Bl:$28),  {Blue}
                                  (Rd:$00; Gr:$28; Bl:$00),  {Green}
                                  (Rd:$00; Gr:$28; Bl:$28),  {Cyan}
                                  (Rd:$28; Gr:$07; Bl:$07),  {Red}
                                  (Rd:$28; Gr:$00; Bl:$28),  {Magenta}
                                  (Rd:$1A; Gr:$1A; Bl:$1A),  {Dark Grey}
                                  (Rd:$00; Gr:$00; Bl:$00)   {Black}        )  );

(*       VGAdefault_w :         {Normal, but white background}
              RECORD
                Sz, Offset : Byte;
                Clr        : ARRAY [0..15] OF RGBrec;
              END
                   = (  Sz:16;  Offset:0;
                        Clr : (
                                  (Rd:$3C; Gr:$3C; Bl:$3C),  {Br White}
                                  (Rd:$00; Gr:$00; Bl:$3C),  {Lt. Blue}
                                  (Rd:$00; Gr:$3C; Bl:$00),  {Lt. Green}
                                  (Rd:$00; Gr:$3C; Bl:$3C),  {Lt. Cyan}
                                  (Rd:$3C; Gr:$00; Bl:$00),  {Lt. Red}
                                  (Rd:$3C; Gr:$00; Bl:$3C),  {Lt. Magenta}
                                  (Rd:$3C; Gr:$3C; Bl:$00),  {Yellow}
                                  (Rd:$2C; Gr:$2C; Bl:$2C),  {Lt Grey}
                                  (Rd:$2C; Gr:$1C; Bl:$00),  {Brown}
                                  (Rd:$00; Gr:$00; Bl:$2C),  {Blue}
                                  (Rd:$00; Gr:$2C; Bl:$00),  {Green}
                                  (Rd:$00; Gr:$2C; Bl:$2C),  {Cyan}
                                  (Rd:$2C; Gr:$00; Bl:$00),  {Red}
                                  (Rd:$2C; Gr:$00; Bl:$2C),  {Magenta}
                                  (Rd:$1C; Gr:$1C; Bl:$1C),  {Dark Grey}
                                  (Rd:$00; Gr:$00; Bl:$00)   {Black}        )  );
*)

       VGAdefault_g :         {Normal, but grey scale, white background}
              RECORD
                Sz, Offset : Byte;
                Clr        : ARRAY [0..15] OF RGBrec;
              END
                   = (  Sz:16;  Offset:0;
                        Clr : (
(*                                 (Rd:$3F; Gr:$3F; Bl:$3F),  {Br White} *)
                                 (Rd:$3C; Gr:$3C; Bl:$3C),  {Br White}
                                 (Rd:$38; Gr:$38; Bl:$38),
                                 (Rd:$34; Gr:$34; Bl:$34),
                                 (Rd:$30; Gr:$30; Bl:$30),
                                 (Rd:$2C; Gr:$2C; Bl:$2C),
                                 (Rd:$28; Gr:$28; Bl:$28),
                                 (Rd:$24; Gr:$24; Bl:$24),
                                 (Rd:$28; Gr:$28; Bl:$28),

                                 (Rd:$18; Gr:$18; Bl:$18),
                                 (Rd:$0C; Gr:$0C; Bl:$0C),
                                 (Rd:$1C; Gr:$1C; Bl:$1C),
                                 (Rd:$2C; Gr:$2C; Bl:$2C),
(*                                 (Rd:$04; Gr:$04; Bl:$04),*)
                                 (Rd:$20; Gr:$20; Bl:$20),
                                 (Rd:$14; Gr:$14; Bl:$14),
                                 (Rd:$24; Gr:$24; Bl:$24),
                                 (Rd:$00; Gr:$00; Bl:$00)   {Black}        )  );

       VGARainbow16 :
             RECORD
               Sz, Offset : Byte;
               Clr        : ARRAY [0..15] OF RGBrec;
             END
                  = (  Sz:16;  Offset:0;
                       Clr : (
                                 (Rd:$00; Gr:$00; Bl:$00),  {Black}

                                 (Rd:$10; Gr:$08; Bl:$24),
                                 (Rd:$10; Gr:$08; Bl:$30),
                                 (Rd:$08; Gr:$08; Bl:$3C),

                                 (Rd:$08; Gr:$20; Bl:$3C),
                                 (Rd:$08; Gr:$28; Bl:$2C),
                                 (Rd:$08; Gr:$2C; Bl:$24),

                                 (Rd:$08; Gr:$34; Bl:$20),
                                 (Rd:$0C; Gr:$3F; Bl:$0C),

                                 (Rd:$24; Gr:$30; Bl:$08),
                                 (Rd:$28; Gr:$1C; Bl:$08),
                                 (Rd:$28; Gr:$10; Bl:$04),
                                 (Rd:$30; Gr:$04; Bl:$04),
                                 (Rd:$3C; Gr:$04; Bl:$04),

                                 (Rd:$3C; Gr:$3C; Bl:$00),  {Yellow}
(*                                 (Rd:$3C; Gr:$3C; Bl:$3C)   {Br White}     )  );*)
                                 (Rd:$3F; Gr:$3F; Bl:$3F)   {Br White}     )  );

       VGARainbow16_w :               {Rainbow, white background}
             RECORD
               Sz, Offset : Byte;
               Clr        : ARRAY [0..15] OF RGBrec;
             END
                  = (  Sz:16;  Offset:0;
                       Clr : (
(*                                 (Rd:$3C; Gr:$3C; Bl:$3C),  {Br White}*)
                                 (Rd:$3F; Gr:$3F; Bl:$3F),  {Br White}

                                 (Rd:$10; Gr:$08; Bl:$24),
                                 (Rd:$10; Gr:$08; Bl:$30),
                                 (Rd:$08; Gr:$08; Bl:$3C),

                                 (Rd:$08; Gr:$20; Bl:$3C),
                                 (Rd:$08; Gr:$28; Bl:$2C),
                                 (Rd:$08; Gr:$2C; Bl:$24),

                                 (Rd:$08; Gr:$34; Bl:$20),
                                 (Rd:$0C; Gr:$3F; Bl:$0C),

                                 (Rd:$24; Gr:$30; Bl:$08),
                                 (Rd:$28; Gr:$1C; Bl:$08),
                                 (Rd:$28; Gr:$10; Bl:$04),
                                 (Rd:$30; Gr:$04; Bl:$04),
                                 (Rd:$3C; Gr:$04; Bl:$04),

                                 (Rd:$1C; Gr:$1C; Bl:$1C),  {Dark Grey}
                                 (Rd:$00; Gr:$00; Bl:$00)   {Black}        )  );

       VGARainbow16_g :               {Rainbow greyscale}
             RECORD
               Sz, Offset : Byte;
               Clr        : ARRAY [0..15] OF RGBrec;
             END
                  = (  Sz:16;  Offset:0;
                       Clr : (
(*                                 (Rd:$3C; Gr:$3C; Bl:$3C),*)
                                 (Rd:$3F; Gr:$3F; Bl:$3F),  {Br White}
                                 (Rd:$38; Gr:$38; Bl:$38),
                                 (Rd:$34; Gr:$34; Bl:$34),
                                 (Rd:$30; Gr:$30; Bl:$30),

                                 (Rd:$2C; Gr:$2C; Bl:$2C),
                                 (Rd:$28; Gr:$28; Bl:$28),
                                 (Rd:$24; Gr:$24; Bl:$24),
                                 (Rd:$20; Gr:$20; Bl:$20),

                                 (Rd:$1C; Gr:$1C; Bl:$1C),
                                 (Rd:$18; Gr:$18; Bl:$18),
                                 (Rd:$14; Gr:$14; Bl:$14),
                                 (Rd:$10; Gr:$10; Bl:$10),

                                 (Rd:$0C; Gr:$0C; Bl:$0C),
                                 (Rd:$08; Gr:$08; Bl:$08),
                                 (Rd:$04; Gr:$04; Bl:$04),
                                 (Rd:$00; Gr:$00; Bl:$00)
                             )   );


       VGARainbow256:
             RECORD
               Sz, Offset : Byte;
               Clr        : ARRAY [0..55] OF RGBrec;
             END
                  = (  Sz:56;  Offset:VGAoffset;
                       Clr : (   (Rd:$00; Gr:$00; Bl:$20),
                                 (Rd:$00; Gr:$00; Bl:$24),
                                 (Rd:$00; Gr:$00; Bl:$28),
                                 (Rd:$00; Gr:$00; Bl:$2C),
                                 (Rd:$00; Gr:$00; Bl:$30),
                                 (Rd:$00; Gr:$00; Bl:$34),
                                 (Rd:$00; Gr:$00; Bl:$38),
                                 (Rd:$00; Gr:$00; Bl:$3C),

                                 (Rd:$00; Gr:$10; Bl:$2C),
                                 (Rd:$00; Gr:$14; Bl:$28),
                                 (Rd:$00; Gr:$18; Bl:$24),
                                 (Rd:$00; Gr:$1C; Bl:$20),
                                 (Rd:$00; Gr:$20; Bl:$1C),
                                 (Rd:$00; Gr:$24; Bl:$18),
                                 (Rd:$00; Gr:$28; Bl:$14),
                                 (Rd:$00; Gr:$2C; Bl:$10),

                                 (Rd:$00; Gr:$20; Bl:$00),
                                 (Rd:$00; Gr:$24; Bl:$00),
                                 (Rd:$00; Gr:$28; Bl:$00),
                                 (Rd:$00; Gr:$2C; Bl:$00),
                                 (Rd:$00; Gr:$30; Bl:$00),
                                 (Rd:$00; Gr:$34; Bl:$00),
                                 (Rd:$00; Gr:$38; Bl:$00),
                                 (Rd:$00; Gr:$3C; Bl:$00),

                                 (Rd:$10; Gr:$2C; Bl:$00),
                                 (Rd:$14; Gr:$28; Bl:$00),
                                 (Rd:$18; Gr:$24; Bl:$00),
                                 (Rd:$1C; Gr:$20; Bl:$00),
                                 (Rd:$20; Gr:$1C; Bl:$00),
                                 (Rd:$24; Gr:$18; Bl:$00),
                                 (Rd:$28; Gr:$14; Bl:$00),
                                 (Rd:$2C; Gr:$10; Bl:$00),

                                 (Rd:$20; Gr:$00; Bl:$00),
                                 (Rd:$24; Gr:$00; Bl:$00),
                                 (Rd:$28; Gr:$00; Bl:$00),
                                 (Rd:$2C; Gr:$00; Bl:$00),
                                 (Rd:$30; Gr:$00; Bl:$00),
                                 (Rd:$34; Gr:$00; Bl:$00),
                                 (Rd:$38; Gr:$00; Bl:$00),
                                 (Rd:$3C; Gr:$00; Bl:$00),

                                 (Rd:$38; Gr:$00; Bl:$04),
                                 (Rd:$34; Gr:$00; Bl:$08),
                                 (Rd:$30; Gr:$00; Bl:$0C),
                                 (Rd:$2C; Gr:$00; Bl:$10),
                                 (Rd:$28; Gr:$00; Bl:$14),
                                 (Rd:$24; Gr:$00; Bl:$18),
                                 (Rd:$20; Gr:$00; Bl:$1C),
                                 (Rd:$1C; Gr:$00; Bl:$20),

                                 (Rd:$20; Gr:$04; Bl:$20),
                                 (Rd:$20; Gr:$08; Bl:$20),
                                 (Rd:$20; Gr:$0C; Bl:$20),
                                 (Rd:$20; Gr:$10; Bl:$20),
                                 (Rd:$20; Gr:$14; Bl:$20),
                                 (Rd:$20; Gr:$18; Bl:$20),
                                 (Rd:$20; Gr:$1C; Bl:$20),
                                 (Rd:$20; Gr:$20; Bl:$20)    )  );

       SVGARainbow256a:
             RECORD
               Sz, Offset : Byte;
               Clr        : ARRAY [0..97] OF RGBrec;
             END
                  = (  Sz:98;  Offset:SVGAoffset;
                       Clr : (
                                 (Rd:$00; Gr:$00; Bl:$02),
                                 (Rd:$00; Gr:$00; Bl:$04),
                                 (Rd:$00; Gr:$00; Bl:$06),
                                 (Rd:$00; Gr:$00; Bl:$08),
                                 (Rd:$00; Gr:$00; Bl:$0A),
                                 (Rd:$00; Gr:$00; Bl:$0C),
                                 (Rd:$00; Gr:$00; Bl:$0E),
                                 (Rd:$00; Gr:$00; Bl:$10),
                                 (Rd:$00; Gr:$00; Bl:$12),
                                 (Rd:$00; Gr:$00; Bl:$14),
                                 (Rd:$00; Gr:$00; Bl:$16),
                                 (Rd:$00; Gr:$00; Bl:$18),
                                 (Rd:$00; Gr:$00; Bl:$1A),
                                 (Rd:$00; Gr:$00; Bl:$1C),
                                 (Rd:$00; Gr:$00; Bl:$1E),

                                 (Rd:$00; Gr:$00; Bl:$20),
                                 (Rd:$00; Gr:$00; Bl:$22),
                                 (Rd:$00; Gr:$00; Bl:$24),
                                 (Rd:$00; Gr:$00; Bl:$26),
                                 (Rd:$00; Gr:$00; Bl:$28),
                                 (Rd:$00; Gr:$00; Bl:$2A),
                                 (Rd:$00; Gr:$00; Bl:$2C),
                                 (Rd:$00; Gr:$00; Bl:$2E),
                                 (Rd:$00; Gr:$00; Bl:$30),
                                 (Rd:$00; Gr:$00; Bl:$32),
                                 (Rd:$00; Gr:$00; Bl:$34),
                                 (Rd:$00; Gr:$00; Bl:$36),
                                 (Rd:$00; Gr:$00; Bl:$38),
                                 (Rd:$00; Gr:$00; Bl:$3A),
                                 (Rd:$00; Gr:$00; Bl:$3C),
                                 (Rd:$00; Gr:$00; Bl:$3E),
                                 (Rd:$00; Gr:$00; Bl:$3F),

                                 (Rd:$00; Gr:$02; Bl:$3F),
                                 (Rd:$00; Gr:$04; Bl:$3E),
                                 (Rd:$00; Gr:$06; Bl:$3C),
                                 (Rd:$00; Gr:$08; Bl:$3A),
                                 (Rd:$00; Gr:$0A; Bl:$38),
                                 (Rd:$00; Gr:$0C; Bl:$36),
                                 (Rd:$00; Gr:$0E; Bl:$34),
                                 (Rd:$00; Gr:$10; Bl:$32),
                                 (Rd:$00; Gr:$12; Bl:$30),
                                 (Rd:$00; Gr:$14; Bl:$2E),
                                 (Rd:$00; Gr:$16; Bl:$2C),
                                 (Rd:$00; Gr:$18; Bl:$2A),
                                 (Rd:$00; Gr:$1A; Bl:$28),
                                 (Rd:$00; Gr:$1C; Bl:$26),
                                 (Rd:$00; Gr:$1E; Bl:$24),
                                 (Rd:$00; Gr:$20; Bl:$22),
                                 (Rd:$00; Gr:$22; Bl:$20),

                                 (Rd:$00; Gr:$24; Bl:$1E),
                                 (Rd:$00; Gr:$26; Bl:$1C),
                                 (Rd:$00; Gr:$28; Bl:$1A),
                                 (Rd:$00; Gr:$2A; Bl:$18),
                                 (Rd:$00; Gr:$2C; Bl:$16),
                                 (Rd:$00; Gr:$2E; Bl:$14),
                                 (Rd:$00; Gr:$30; Bl:$12),
                                 (Rd:$00; Gr:$32; Bl:$10),
                                 (Rd:$00; Gr:$34; Bl:$0E),
                                 (Rd:$00; Gr:$36; Bl:$0C),
                                 (Rd:$00; Gr:$38; Bl:$0A),
                                 (Rd:$00; Gr:$3A; Bl:$08),
                                 (Rd:$00; Gr:$3C; Bl:$06),
                                 (Rd:$00; Gr:$3E; Bl:$04),
                                 (Rd:$00; Gr:$3F; Bl:$02),
                                 (Rd:$00; Gr:$3F; Bl:$00),


                                 (Rd:$02; Gr:$3F; Bl:$00),
                                 (Rd:$04; Gr:$3E; Bl:$00),
                                 (Rd:$06; Gr:$3C; Bl:$00),
                                 (Rd:$08; Gr:$3A; Bl:$00),
                                 (Rd:$0A; Gr:$38; Bl:$00),
                                 (Rd:$0C; Gr:$36; Bl:$00),
                                 (Rd:$0E; Gr:$34; Bl:$00),
                                 (Rd:$10; Gr:$32; Bl:$00),
                                 (Rd:$12; Gr:$30; Bl:$00),
                                 (Rd:$14; Gr:$2E; Bl:$00),
                                 (Rd:$16; Gr:$2C; Bl:$00),
                                 (Rd:$18; Gr:$2A; Bl:$00),
                                 (Rd:$1A; Gr:$28; Bl:$00),
                                 (Rd:$1C; Gr:$26; Bl:$00),
                                 (Rd:$1E; Gr:$24; Bl:$00),
                                 (Rd:$20; Gr:$22; Bl:$00),

                                 (Rd:$22; Gr:$20; Bl:$00),
                                 (Rd:$24; Gr:$1E; Bl:$00),
                                 (Rd:$26; Gr:$1C; Bl:$00),
                                 (Rd:$28; Gr:$1A; Bl:$00),
                                 (Rd:$2A; Gr:$18; Bl:$00),
                                 (Rd:$2C; Gr:$16; Bl:$00),
                                 (Rd:$2E; Gr:$14; Bl:$00),
                                 (Rd:$30; Gr:$12; Bl:$00),
                                 (Rd:$32; Gr:$10; Bl:$00),
                                 (Rd:$34; Gr:$0E; Bl:$00),
                                 (Rd:$36; Gr:$0C; Bl:$00),
                                 (Rd:$38; Gr:$0A; Bl:$00),
                                 (Rd:$3A; Gr:$08; Bl:$00),
                                 (Rd:$3C; Gr:$06; Bl:$00),
                                 (Rd:$3E; Gr:$04; Bl:$00),
                                 (Rd:$3F; Gr:$02; Bl:$00),


                                 (Rd:$3F; Gr:$00; Bl:$00)

                                                              )  );


       SVGARainbow256b:
             RECORD
               Sz, Offset : Byte;
               Clr        : ARRAY [0..72] OF RGBrec;
             END
                  = (  Sz:73;  Offset:SVGAoffset;
                       Clr : (
                                 (Rd:$04; Gr:$04; Bl:$0C),
                                 (Rd:$04; Gr:$04; Bl:$14),
                                 (Rd:$04; Gr:$04; Bl:$18),
                                 (Rd:$04; Gr:$04; Bl:$1C),

                                 (Rd:$02; Gr:$02; Bl:$20),
                                 (Rd:$02; Gr:$02; Bl:$24),
                                 (Rd:$00; Gr:$00; Bl:$28),
                                 (Rd:$00; Gr:$00; Bl:$2A),
                                 (Rd:$00; Gr:$00; Bl:$2E),
                                 (Rd:$00; Gr:$00; Bl:$34),
                                 (Rd:$00; Gr:$00; Bl:$38),
                                 (Rd:$00; Gr:$00; Bl:$3C),

                                 (Rd:$00; Gr:$00; Bl:$3F),


                                 (Rd:$00; Gr:$04; Bl:$3C),
                                 (Rd:$00; Gr:$08; Bl:$38),
                                 (Rd:$00; Gr:$0C; Bl:$34),
                                 (Rd:$00; Gr:$10; Bl:$30),
                                 (Rd:$00; Gr:$12; Bl:$2E),
                                 (Rd:$00; Gr:$14; Bl:$2C),
                                 (Rd:$00; Gr:$16; Bl:$2A),
                                 (Rd:$00; Gr:$18; Bl:$28),
                                 (Rd:$00; Gr:$1A; Bl:$26),
                                 (Rd:$00; Gr:$1C; Bl:$24),
                                 (Rd:$00; Gr:$1D; Bl:$23),
                                 (Rd:$00; Gr:$1E; Bl:$22),
                                 (Rd:$00; Gr:$1F; Bl:$21),
                                 (Rd:$00; Gr:$20; Bl:$21),

                                 (Rd:$00; Gr:$20; Bl:$20),
                                 (Rd:$00; Gr:$21; Bl:$20),
                                 (Rd:$00; Gr:$21; Bl:$1F),
                                 (Rd:$00; Gr:$22; Bl:$1E),
                                 (Rd:$00; Gr:$23; Bl:$1D),
                                 (Rd:$00; Gr:$24; Bl:$1C),
                                 (Rd:$00; Gr:$26; Bl:$1A),
                                 (Rd:$00; Gr:$28; Bl:$18),
                                 (Rd:$00; Gr:$2A; Bl:$16),
                                 (Rd:$00; Gr:$2C; Bl:$14),
                                 (Rd:$00; Gr:$2E; Bl:$12),
                                 (Rd:$00; Gr:$30; Bl:$10),
                                 (Rd:$00; Gr:$34; Bl:$0C),
                                 (Rd:$00; Gr:$38; Bl:$08),
                                 (Rd:$00; Gr:$3C; Bl:$04),

                                 (Rd:$00; Gr:$3F; Bl:$00),


                                 (Rd:$04; Gr:$3C; Bl:$00),
                                 (Rd:$08; Gr:$38; Bl:$00),
                                 (Rd:$0C; Gr:$34; Bl:$00),
                                 (Rd:$10; Gr:$30; Bl:$00),
                                 (Rd:$12; Gr:$2E; Bl:$00),
                                 (Rd:$14; Gr:$2C; Bl:$00),
                                 (Rd:$16; Gr:$2A; Bl:$00),
                                 (Rd:$18; Gr:$28; Bl:$00),
                                 (Rd:$1A; Gr:$26; Bl:$00),
                                 (Rd:$1C; Gr:$24; Bl:$00),
                                 (Rd:$1D; Gr:$23; Bl:$00),
                                 (Rd:$1E; Gr:$22; Bl:$00),
                                 (Rd:$1F; Gr:$21; Bl:$00),
                                 (Rd:$20; Gr:$21; Bl:$00),

                                 (Rd:$20; Gr:$20; Bl:$00),
                                 (Rd:$21; Gr:$20; Bl:$00),
                                 (Rd:$21; Gr:$1F; Bl:$00),
                                 (Rd:$22; Gr:$1E; Bl:$00),
                                 (Rd:$23; Gr:$1D; Bl:$00),
                                 (Rd:$24; Gr:$1C; Bl:$00),
                                 (Rd:$26; Gr:$1A; Bl:$00),
                                 (Rd:$28; Gr:$18; Bl:$00),
                                 (Rd:$2A; Gr:$16; Bl:$00),
                                 (Rd:$2C; Gr:$14; Bl:$00),
                                 (Rd:$2E; Gr:$12; Bl:$00),
                                 (Rd:$30; Gr:$10; Bl:$00),
                                 (Rd:$34; Gr:$0C; Bl:$00),
                                 (Rd:$38; Gr:$08; Bl:$00),
                                 (Rd:$3C; Gr:$04; Bl:$00),

                                 (Rd:$3F; Gr:$00; Bl:$00)

                                                              )  );


       SVGARainbow256c:
             RECORD
               Sz, Offset : Byte;
               Clr        : ARRAY [0..81] OF RGBrec;
             END
                  = (  Sz:82;  Offset:SVGAoffset;
                       Clr : (
                                 (Rd:$00; Gr:$10; Bl:$3F),
                                 (Rd:$00; Gr:$0E; Bl:$3F),
                                 (Rd:$00; Gr:$0C; Bl:$3F),
                                 (Rd:$00; Gr:$0A; Bl:$3F),
                                 (Rd:$00; Gr:$08; Bl:$3F),
                                 (Rd:$00; Gr:$06; Bl:$3F),
                                 (Rd:$00; Gr:$04; Bl:$3F),
                                 (Rd:$00; Gr:$02; Bl:$3F),
                                 (Rd:$00; Gr:$00; Bl:$3F),
                                 (Rd:$00; Gr:$00; Bl:$3E),
                                 (Rd:$00; Gr:$00; Bl:$3C),
                                 (Rd:$00; Gr:$00; Bl:$3A),
                                 (Rd:$00; Gr:$00; Bl:$38),
                                 (Rd:$00; Gr:$00; Bl:$36),
                                 (Rd:$00; Gr:$00; Bl:$34),
                                 (Rd:$00; Gr:$00; Bl:$32),
                                 (Rd:$00; Gr:$00; Bl:$30),
                                 (Rd:$00; Gr:$00; Bl:$2E),
                                 (Rd:$00; Gr:$00; Bl:$2C),
                                 (Rd:$00; Gr:$00; Bl:$2A),
                                 (Rd:$00; Gr:$00; Bl:$28),
                                 (Rd:$00; Gr:$00; Bl:$26),
                                 (Rd:$00; Gr:$00; Bl:$24),
                                 (Rd:$00; Gr:$00; Bl:$22),
                                 (Rd:$00; Gr:$00; Bl:$20),
                                 (Rd:$00; Gr:$00; Bl:$1E),
                                 (Rd:$00; Gr:$00; Bl:$1C),
                                 (Rd:$00; Gr:$00; Bl:$1A),
                                 (Rd:$00; Gr:$00; Bl:$18),
                                 (Rd:$00; Gr:$00; Bl:$16),
                                 (Rd:$00; Gr:$00; Bl:$14),
                                 (Rd:$00; Gr:$00; Bl:$12),
                                 (Rd:$00; Gr:$00; Bl:$10),
                                 (Rd:$00; Gr:$00; Bl:$0E),
                                 (Rd:$00; Gr:$00; Bl:$0C),
                                 (Rd:$00; Gr:$00; Bl:$0A),
                                 (Rd:$00; Gr:$00; Bl:$08),
                                 (Rd:$00; Gr:$00; Bl:$06),
                                 (Rd:$00; Gr:$00; Bl:$04),
                                 (Rd:$00; Gr:$00; Bl:$02),
                                 (Rd:$00; Gr:$00; Bl:$00),

                                 (Rd:$00; Gr:$00; Bl:$00),
                                 (Rd:$02; Gr:$00; Bl:$00),
                                 (Rd:$04; Gr:$00; Bl:$00),
                                 (Rd:$06; Gr:$00; Bl:$00),
                                 (Rd:$08; Gr:$00; Bl:$00),
                                 (Rd:$0A; Gr:$00; Bl:$00),
                                 (Rd:$0C; Gr:$00; Bl:$00),
                                 (Rd:$0E; Gr:$00; Bl:$00),
                                 (Rd:$10; Gr:$00; Bl:$00),
                                 (Rd:$12; Gr:$00; Bl:$00),
                                 (Rd:$14; Gr:$00; Bl:$00),
                                 (Rd:$16; Gr:$00; Bl:$00),
                                 (Rd:$18; Gr:$00; Bl:$00),
                                 (Rd:$1A; Gr:$00; Bl:$00),
                                 (Rd:$1C; Gr:$00; Bl:$00),
                                 (Rd:$1E; Gr:$00; Bl:$00),
                                 (Rd:$20; Gr:$00; Bl:$00),
                                 (Rd:$22; Gr:$00; Bl:$00),
                                 (Rd:$24; Gr:$00; Bl:$00),
                                 (Rd:$26; Gr:$00; Bl:$00),
                                 (Rd:$28; Gr:$00; Bl:$00),
                                 (Rd:$2A; Gr:$00; Bl:$00),
                                 (Rd:$2C; Gr:$00; Bl:$00),
                                 (Rd:$2E; Gr:$00; Bl:$00),
                                 (Rd:$30; Gr:$00; Bl:$00),
                                 (Rd:$32; Gr:$00; Bl:$00),
                                 (Rd:$34; Gr:$00; Bl:$00),
                                 (Rd:$36; Gr:$00; Bl:$00),
                                 (Rd:$38; Gr:$00; Bl:$00),
                                 (Rd:$3A; Gr:$00; Bl:$00),
                                 (Rd:$3C; Gr:$00; Bl:$00),
                                 (Rd:$3E; Gr:$00; Bl:$00),
                                 (Rd:$3F; Gr:$00; Bl:$00),
                                 (Rd:$3F; Gr:$02; Bl:$00),
                                 (Rd:$3F; Gr:$04; Bl:$00),
                                 (Rd:$3F; Gr:$06; Bl:$00),
                                 (Rd:$3F; Gr:$08; Bl:$00),
                                 (Rd:$3F; Gr:$0A; Bl:$00),
                                 (Rd:$3F; Gr:$0C; Bl:$00),
                                 (Rd:$3F; Gr:$0E; Bl:$00),
                                 (Rd:$3F; Gr:$10; Bl:$00)

                                                              )  );

     TYPE
       Ds_Window   = RECORD
                          St, Fin, Sz : Pixels; {Start & Finish Pixel co-ordinates & Size}
                        END;

     VAR
       SeismWind              : ARRAY [1..MaxSeism] OF
                                  ARRAY [1..3]  OF Ds_Window;
       QtySeism, QtySWind     : Word;
       SingleWind             : Boolean;
       GraphWind, MsgWind,
          LegWind             : Ds_Window;
       GraphicsWind, HzTWind,
          VtTWind, HdTWind    : Ds_Window;
       SnapWindMx             : Ds_Window;
       MenuWind               : Ds_Window;


     VAR
       VGA256set              : Boolean;
       Rainbow                : Boolean;     {Graded pallette, rainbow effect}
       ColOffset              : Word;
       MaxColours             : Word;
       Aspect                 : Single;
       ScrnAsp                : Single;       {Physical i.t.o. screen size}
       Hpix, Vpix             : Integer;
       GraphDriver, GraphMode : Integer;
       Hscale, Vscale         : Word;
       Hinset                 : Word;
       FontSz                 : Word;
       Ds_VectSc              : Single;


   {Extensions to the BGI}

     PROCEDURE LineTo2 (Hz_end, Vt_end : Integer);
     PROCEDURE Line2 (Hz_st, Vt_st, Hz_end, Vt_end : Integer);

     PROCEDURE ResetViewPort;

     PROCEDURE SetLineType (Lnum : Integer);

     PROCEDURE ResetLineStyle;

     PROCEDURE Ds_ResetTextStyle;


   {Initialization of Graphics System}

     PROCEDURE SetAllRGB (Pal : RGBpalPtr);

     PROCEDURE Ds_256Palette;

     PROCEDURE Ds_ResetPalette;

     PROCEDURE Ds_SetPalette;

     PROCEDURE Ds_LinkDrivers;

     PROCEDURE Ds_InitVid;


   {Windowing Routines}

     PROCEDURE Ds_SetWindow (w: Ds_Window);

     PROCEDURE Ds_ClearWindow (w: Ds_Window);

     PROCEDURE Ds_DrawWindow (w: Ds_Window);

     PROCEDURE Ds_InitWindows;

     PROCEDURE Ds_TestWindows;

     PROCEDURE Ds_CalcSeismWind;

     PROCEDURE Ds_ClearSeism;


   {Message Routines}

     FUNCTION Ds_ReadKey: Char;

     PROCEDURE Ds_OutText (S : String; Col: Word);

     PROCEDURE Ds_WaitMessage (S : String;  VAR Ch : Char);

     PROCEDURE Ds_AddMessage (Snum, Sqty : Word; S : String);

     PROCEDURE Ds_WaitResponse (VAR Ch : Char);

     PROCEDURE Ds_GetInput (  Msg : String;
                            VAR S : String);


   {Macro Routines for graphing}

     PROCEDURE Ds_CentreBar (Hz_st, Vt_st, Len, Ang: Single);

     PROCEDURE Ds_DrawArrow (Hz_tip, Vt_tip, Hz_len, Vt_len: Single);


  IMPLEMENTATION

     USES
       BGIdriv, BGIfon1, BGIfon2, BGIfon3, SVGA, Drivers, Funct;


   {Extensions to the BGI}

       PROCEDURE Line2 (Hz_st, Vt_st, Hz_end, Vt_end : Integer);
         BEGIN
           Line (Hz_st, Vt_st, Hz_end, Vt_end);
           Line (Hz_st-1, Vt_st-1, Hz_end-1, Vt_end-1);
         END; {Line2}

       PROCEDURE LineTo2 (Hz_end, Vt_end : Integer);
         VAR
           i,j : Integer;
         BEGIN
           i:=GetX; j:=GetY;
           Line (i, j, Hz_end, Vt_end);
{           Line (i-1, j-1, Hz_end-1, Vt_end-1);}
           MoveTo(Hz_end,Vt_end);
         END; {Pl_Line2}


       PROCEDURE ResetViewPort;
         BEGIN
           SetViewPort (0,0,Hpix-1, Vpix-1, ClipOn);
         END; {ResetViewPort}

       PROCEDURE SetLineType (Lnum : Integer);
         BEGIN
           CASE Lnum OF
              1 : SetLineStyle (SolidLn,  0, NormWidth);
              2 : SetLineStyle (DashedLn, 0, NormWidth);
              3 : SetLineStyle (DottedLn, 0, NormWidth);
              4 : SetLineStyle (CenterLn, 0, NormWidth);
              5 : SetLineStyle (SolidLn,  0, ThickWidth);
              6 : SetLineStyle (DashedLn, 0, ThickWidth);
              ELSE SetLineStyle (SolidLn,  0, NormWidth);
           END; {Case}
         END; {SetLineType}

       PROCEDURE ResetLineStyle;
         BEGIN
           SetLineStyle (SolidLn, 0, NormWidth);
         END; {ResetLineStyle}

       PROCEDURE Ds_ResetTextStyle;
         BEGIN
           SetTextStyle (DefaultFont, Horizdir, 1);
         END; {ResetLineStyle}


  {Graphics Initialization}

     PROCEDURE SetAllRGB (Pal : RGBpalPtr);
       VAR
         Col : Integer;
       BEGIN
         WITH Pal^ DO
           FOR Col := Offset TO (Sz+Offset-1) DO
             BEGIN
               WITH Clr[Col-Offset] DO
                 SetRGBPalette (Col, Rd, Gr, Bl);
               IF (GraphDriver IN [VGA,SVGA256])
                  THEN SetPalette (Col,Col);
             END; {For}
         ColOffSet := Pal^.Offset;
       END; {SetAllRGB}

     PROCEDURE Ds_256Palette;
       BEGIN
         IF (GraphDriver IN [IBM8514,SVGA256])
            THEN BEGIN
{                   SetAllRGB (@VGARainBow16);}
                   SetAllRGB (@VGAdefault);
                   SetAllRGB (@VGARainBow256);
                   SetAllRGB (@SVGARainBow256b);
                 END;
       END; {Ds_256Palette}

     PROCEDURE Ds_ResetPalette;
       VAR
         OrigPalette : PaletteType;
       BEGIN
         Rainbow := False;
         CASE GraphDriver OF
           IBM8514,SVGA256,SVGA16,
           VGA     : BEGIN
                       CASE (sets.ColScale) OF
                          Col_blackb: SetAllRGB (@VGAdefault);
                          Col_whiteb: SetAllRGB (@VGAdefault_w);
                          Col_grey:   SetAllRGB (@VGAdefault_g);
                       END; {Case}
                       ColOffSet := EGAoffset;
                     END; {VGA default Palette}
           EGA     : BEGIN
                       SetAllPalette (EGAdefault);
                       ColOffSet := EGAoffset;
                     END; {EGA}
         END; {Case GraphDriver}
       END; {Ds_ResetPalette}

     PROCEDURE Ds_SetPalette;
       BEGIN
         Rainbow := True;
         IF (sets.col256)
            THEN CASE GraphDriver OF
                   IBM8514 : BEGIN
{                               SetAllRGB (@VGARainBow256);}
                               ColOffSet := VGAoffset;
                             END; {IBM8514 and VGA 256 colours}
                   SVGA256 : BEGIN
{                               SetAllRGB (@SVGARainBow256b);}
                               ColOffSet := SVGAoffset;
                             END; {SVGA 256 colours}
                   SVGA16,
                   VGA     : BEGIN
                               CASE (sets.ColScale) OF
                                  Col_blackb: SetAllRGB (@VGARainBow16);
                                  Col_whiteb: SetAllRGB (@VGARainBow16_w);
                                  Col_grey:   SetAllRGB (@VGARainBow16_g);
                               END; {Case}
                               ColOffSet := EGAoffset;
                             END; {16 colours - HiRes VGA}
                   EGA     : BEGIN
                               SetAllPalette (EGARainBow);
                               ColOffSet := EGAoffset;
                             END; {EGA}
                 END {Case GraphDriver}
            ELSE CASE GraphDriver OF
                   IBM8514,SVGA256,SVGA16,
                   VGA     : BEGIN
                               CASE (sets.ColScale) OF
                                  Col_blackb: SetAllRGB (@VGARainBow16);
                                  Col_whiteb: SetAllRGB (@VGARainBow16_w);
                                  Col_grey:   SetAllRGB (@VGARainBow16_g);
                               END; {Case}
                               ColOffSet := EGAoffset;
                             END; {16 colours - HiRes VGA}
                   EGA     : BEGIN
                               SetAllPalette (EGARainBow);
                               ColOffSet := EGAoffset;
                             END; {EGA}
                 END; {Case GraphDriver}
       END; {Ds_SetPalette}


     {$F+}
     FUNCTION DetectVGA256 : integer;
       {Detects VGA or MCGA video cards}
       VAR
         DetectedDriver : integer;
         SuggestedMode  : integer;
       BEGIN
         DetectGraph(DetectedDriver, SuggestedMode);
         IF ( (DetectedDriver<>VGA) AND (DetectedDriver<>MCGA) )
            THEN DetectVGA256 := grError { Couldn't detect hardware }
            ELSE CASE Sets.VidMode OF
                   vga256  : DetectVGA256 := 0; {Default mode}
                   ELSE DetectVGA256 := grError;
                 END;
       END; {DetectVGA256}
     {$F-}

     {$F+}
       FUNCTION DetectSVGA256 : Integer;
       VAR
         DetectedDriver : integer;
         SuggestedMode  : integer;
         Vid : Integer;
       BEGIN
         DetectGraph(DetectedDriver, SuggestedMode);
         IF ( (DetectedDriver<>VGA) AND (DetectedDriver<>MCGA) )
            THEN DetectSVGA256 := grError { Couldn't detect hardware }
            ELSE CASE Sets.VidMode OF
                   svga640  : DetectSVGA256 := 2;
                   svga800  : DetectSVGA256 := 3;
                   svga1024 : DetectSVGA256 := 4;
                   ELSE DetectSVGA256 := grError;
                 END;
       END;
     {$F-}

(*     {$F+}
       FUNCTION DetectSVGA16 : Integer;
       VAR
         DetectedDriver : integer;
         SuggestedMode  : integer;
         Vid : Integer;
       BEGIN
         DetectGraph(DetectedDriver, SuggestedMode);
         IF ( (DetectedDriver<>VGA) AND (DetectedDriver<>MCGA) )
            THEN DetectSVGA16 := grError { Couldn't detect hardware }
            ELSE CASE Sets.VidMode OF
                   svga16_800  : DetectSVGA16 := 4;
                   svga16_1024 : DetectSVGA16 := 5;
                   ELSE DetectSVGA16 := grError;
                 END;
          DetectSVGA16 := grError;
       END;
     {$F-}
*)

     PROCEDURE Ds_LinkDrivers;
       PROCEDURE Abort(Msg : string);
         BEGIN
           Writeln(Msg, ': ', GraphErrorMsg(GraphResult));
           Halt(1);
         END; {Abort}

       BEGIN
         GraphDriver := InstallUserDriver('VGA256', @DetectVGA256);
         GraphDriver := InstallUserDriver('SVGA256',@DetectSVGA256);
{         GraphDriver := InstallUserDriver('SVGA16',@DetectSVGA16);}
         { Register all required drivers }
         if RegisterBGIdriver(@EGAVGADriverProc) < 0 then
           Abort('EGA/VGA');
         if RegisterBGIdriver(@VGA256DriverProc) < 0 then
           Abort('VGA256');
         if RegisterBGIdriver(@SVGA256_Driver) < 0 then
           Abort('SVGA256');
{         if RegisterBGIdriver(@SVGA16_Driver) < 0 then
           Abort('SVGA16');}
(*         if RegisterBGIdriver(@CGADriverProc) < 0 then
           Abort('CGA');
         if RegisterBGIdriver(@HercDriverProc) < 0 then
           Abort('Herc');
         if RegisterBGIdriver(@ATTDriverProc) < 0 then
           Abort('AT&T');
         if RegisterBGIdriver(@PC3270DriverProc) < 0 then
           Abort('PC 3270');
         if RegisterBGIdriver(@IBM8514DriverProc) < 0 then
           Abort('IBM8514');                               *)

         { Register all the fonts }
         if RegisterBGIfont(@SmallFontProc) < 0 then
           Abort('Small');
(*         if RegisterBGIfont(@TriplexFontProc) < 0 then
           Abort('Triplex');
         if RegisterBGIfont(@SansSerifFontProc) < 0 then
           Abort('SansSerif');
         if RegisterBGIfont(@GothicFontProc) < 0 then
           Abort('Gothic');
         if RegisterBGIfont(@ScriFontProc) < 0 then
           Abort('Script');
         if RegisterBGIfont(@SimpFontProc) < 0 then
           Abort('Simp');
         if RegisterBGIfont(@TscrFontProc) < 0 then
           Abort('Tscr');
         if RegisterBGIfont(@LcomFontProc) < 0 then
           Abort('Lcom');
         if RegisterBGIfont(@EuroFontProc) < 0 then
           Abort('Euro');
         if RegisterBGIfont(@BoldFontProc) < 0 then
           Abort('Bold');                             *)
       END; {Ds_LinkDrivers}

     PROCEDURE Ds_InitVid;
       VAR
         Hz_aspect, Vt_aspect : Word;
         Err                  : Integer;
       BEGIN
         GraphDriver := Detect;
         Err         := GraphResult;
         InitGraph   (GraphDriver, GraphMode, '');
         Err := GraphResult;
         Hpix    := GetMaxX + 1;
         Vpix    := GetMaxY + 1;
         MaxColours := GetMaxColor+1 - 2;  { Get the maximum allowable drawing color, subtract background and Black}
         GetAspectRatio  (Hz_aspect, Vt_aspect);
         Aspect  := Hz_aspect / Vt_aspect;
         IF (Graphdriver = SVGA256)
            THEN Aspect  := 1;
         ScrnAsp := Aspect * Hpix / Vpix;
         Vscale  := Vpix - 3;
         Hscale  := TRUNC (Vscale / Aspect);
         Hinset  := TRUNC (Hscale * 0.015);
{         FontSz  := ROUND (sqrt(Vpix)/3-2.5);}
         FontSz  := ROUND (sqrt(Vpix)/3-2.5);
         SetTextStyle (SmallFont, HorizDir, FontSz);
         CASE GraphDriver OF
           IBM8514 : Sets.VidMode := vga256;
           SVGA256 : CASE Graphmode OF
                       2 : Sets.VidMode := svga640;
                       3 : Sets.VidMode := svga800;
                       4 : Sets.VidMode := svga1024;
                       ELSE Sets.VidMode := svga640;
                     END; {SVGA modes}
{           SVGA16 : CASE Graphmode OF
                       4 : Sets.VidMode := svga16_800;
                       5 : Sets.VidMode := svga16_1024;
                       ELSE Sets.VidMode := svga16_1024;
                     END; {SVGA modes}
           ELSE Sets.VidMode := std16;
         END; {Case GraphDriver}
         Sets.col256 := Sets.col256 AND (sets.VidMode IN [vga256,svga640,svga800,svga1024]);
       END; {Ds_InitVid}


    {Windowing Routines}

       PROCEDURE Ds_SetWindow (w: Ds_Window);
         BEGIN
           WITH w DO
             SetViewPort (St[1], St[2], Fin[1], Fin[2], ClipOn);
         END; {Ds_SetWindow}

       PROCEDURE Ds_ClearWindow (w: Ds_Window);
         BEGIN
           WITH w DO
             SetViewPort (St[1], St[2], Fin[1], Fin[2], ClipOn);
           ClearViewPort;
           SetViewPort (0,0,Hpix-1, Vpix-1, ClipOn);
         END; {Ds_ClearWindow}

       PROCEDURE Ds_DrawWindow (w: Ds_Window);
         BEGIN
           ResetViewPort;
           WITH w DO
             RECTANGLE (St[1]-1, St[2]-1, Fin[1]+1, Fin[2]+1);
           Ds_SetWindow (w);
         END; {Ds_DrawWindow}


       PROCEDURE Ds_InitWindows;
         VAR
           Vsz, Voff, Hsz, Hoff : Integer;
         BEGIN
           Vsz  := (Vpix-2);
           Hsz  := TRUNC (Vsz / Aspect);
           Voff := Vsz DIV 15;
           Hoff := Hsz DIV 25;
           WITH GraphWind DO
             BEGIN
               St[1]  := 0;  Sz[1]  := Hsz+2;  Fin[1] := St[1]+Sz[1]-1;
               St[2]  := 0;  Sz[2]  := Vsz+2;  Fin[2] := St[2]+Sz[2]-1;
             END; {With GraphWind}
           WITH MsgWind DO
             BEGIN
               St[1]  := GraphWind.Fin[1]+10;  Sz[1]  := Hpix-2-St[1]+1;  Fin[1] := Hpix-2;
               St[2]  := GraphWind.St[2]+1;    Sz[2]  := Vsz DIV 5;       Fin[2] := St[2]+Sz[2]-1;
             END; {With MsgWind}
           WITH LegWind DO
             BEGIN
               St[1]  := MsgWind.St[1];       Sz[1]  := MsgWind.Sz[1];;  Fin[1] := MsgWind.Fin[1];;
               St[2]  := MsgWind.Fin[2] + 2;  Sz[2]  := Vsz-St[2]+1;     Fin[2] := St[2]+Sz[2]-1;
             END; {With LegWind}

           WITH MenuWind DO
             BEGIN
               St[1]  := 0;  Sz[1]  := Hpix;         Fin[1] := Hpix-1;
               St[2]  := 0;  Sz[2]  := Vsz DIV 20;   Fin[2] := St[2]+Sz[2]-1;
             END; {With MenuWind}

           {Graphics and Title windows, dependent on GraphWind}
             WITH HdTWind DO
               BEGIN
                 St[1]  := GraphWind.St[1]+Hoff+1;   Sz[1]  := Hsz-Hoff;       Fin[1] := St[1]+Sz[1]-1;
                 St[2]  := GraphWind.St[2]+1;        Sz[2]  := Voff;           Fin[2] := St[2]+Sz[2]-1;
               END; {With HdTWind}
             WITH HzTWind DO
               BEGIN
                 St[1]  := HdTWind.St[1];            Sz[1]  := HdTWind.Sz[1];  Fin[1] := HdTWind.Fin[1];
                 St[2]  := GraphWind.St[2]+Vsz-Voff; Sz[2]  := Voff;           Fin[2] := St[2]+Sz[2]-1;
               END; {With HzTWind}
             WITH VtTWind DO
               BEGIN
                 St[1]  := GraphWind.St[1]+1;        Sz[1]  := Hoff;           Fin[1] := St[1]+Sz[1]-1;
                 St[2]  := GraphWind.St[2]+Voff+1;   Sz[2]  := Vsz-Voff-Voff;  Fin[2] := St[2]+Sz[2]-1;
               END; {With VtTWind}
             WITH GraphicsWind DO
               BEGIN
                 St[1]  := VtTWind.Fin[1]+1;    Sz[1] := (Hsz-Hoff);       Fin[1] := St[1]+Sz[1]-1;
                 St[2]  := HdTWind.Fin[2]+1;    Sz[2] := (Vsz-Voff-Voff);  Fin[2] := St[2]+Sz[2]-1;
               END; {With GraphicsWind}

               {Snapshot window dependent on Graphics window}
               WITH SnapWindMx DO
                 BEGIN
                   St[1]  := GraphicsWind.St[1]+1;  Sz[1] := GraphicsWind.Sz[1]-2;   Fin[1] := St[1]+Sz[1]-1;
                   St[2]  := GraphicsWind.St[2]+10; Sz[2] := GraphicsWind.Sz[2]-20;  Fin[2] := St[2]+Sz[2]-1;
                 END; {With SnapWindMx}
         END;  {Ds_InitWindows}


       PROCEDURE Ds_TestWindows;
         VAR
           ch : char;
         BEGIN
           SetBkColor (DsCol_Bgrnd);
           SetColor (DsCol_Fgrnd);
           ResetViewport;
           ClearViewport;
           Ds_DrawWindow (GraphWind);
           Ds_DrawWindow (MsgWind);
           Ds_DrawWindow (LegWind);
           Ds_DrawWindow (MenuWind);
           ch := ReadKey;
           Ds_DrawWindow (GraphicsWind);
           Ds_DrawWindow (HdtWind);
           Ds_DrawWindow (HztWind);
           Ds_DrawWindow (VtTWind);
           ch := ReadKey;
           Ds_DrawWindow (SnapWindMx);
           ch := ReadKey;
           ClearViewport;
         END; {Testwindows}


       PROCEDURE Ds_CalcSeismWind;
         CONST
           HzRat : ARRAY [1..6] OF Single = (0.0, 0.10, 0.11, 0.850, 0.86, 1.0);
           VtRat : ARRAY [1..2] OF Single = (0.04, 0.96);
{           HzRat : ARRAY [1..6] OF Single = (0.0, 0.24, 0.245, 0.855, 0.86, 1.0);
           VtRat : ARRAY [1..2] OF Single = (0.04, 0.96);}
         VAR
           i,j   : Word;
           Hz    : ARRAY [1..6] OF Word;
           Vt    : ARRAY [1..2] OF Word;
           VSize : Single;
           Voff  : Word;
         BEGIN
           Ds_ClearWindow (GraphWind);
           WITH SnapWindMx DO
             BEGIN
               FOR i := 1 TO 6 DO
                 Hz[i] := St[1] + TRUNC (Sz[1]*HzRat[i]);
               VSize := Sz[2] / QtySWind;
               Voff  := St[2];
             END; {With}
           FOR i := 1 TO QtySWind DO
             BEGIN
               Vt[1] := TRUNC ( (i-1)*Vsize + VtRat[1]*Vsize) + Voff;
               Vt[2] := TRUNC ( (i-1)*Vsize + VtRat[2]*Vsize) + Voff;
               FOR j := 1 TO 3 DO
                 WITH SeismWind[i,j] DO
                   BEGIN
                     St[1]  := Hz[j*2-1];   Fin[1] := Hz[j*2];  Sz[1]  := Fin[1] - St[1] + 1;
                     St[2]  := Vt[1];       Fin[2] := Vt[2];    Sz[2]  := Fin[2] - St[2] + 1;
                   END; {With}
             END; {For}
           IF (SingleWind)
              THEN FOR i := 2 TO QtySeism DO
                     SeismWind[i] := SeismWind[1];
(*           SetColor (DsCol_Axes);
           Ds_DrawWindow (MsgWind);
           Ds_DrawWindow (LegWind);*)
         END;  {Ds_CalcSeismWind}

       PROCEDURE Ds_ClearSeism;
         VAR
           i,j   : Word;
         BEGIN
           FOR i := 1 TO QtySeism DO
             FOR j := 1 TO 3 DO
               Ds_ClearWindow (SeismWind[i,j]);
         END;  {Ds_ClearSeism}


  {Message Display}

     FUNCTION Ds_ReadKey: Char;
       VAR Ch, Ch2 : Char;
       BEGIN {Readkey, but clears buffer if full}
         Ch := ReadKey;
         IF NOT (Ch IN [#0,'0'..'9']) THEN
            WHILE (Keypressed) DO  Ch2 := ReadKey;
         Ds_ReadKey := Ch;
       END;  {SetupGraph}

     PROCEDURE Ds_OutText (S : String;  Col: Word);
       CONST
         ch1 = '´'; {Open Brackets, CHR(180) }
         ch2 = 'Ã'; {Open Brackets, CHR(195) }
       VAR
         Pos1, Pos2 : Byte;
       BEGIN
{         SetColor (Col);}
         SetTextJustify (LeftText, CenterText);
         WHILE (S <> '') DO
           BEGIN
             Pos1 := Pos (Ch1, S);
             Delete (S, Pos1, 1);
             Pos2 := Pos (Ch2, S);
             Delete (S, Pos2, 1);
             IF (Pos1 = 0)
                THEN BEGIN
                       SetColor (Col);
                       OutText (S);
                       S := '';
                     END
                ELSE BEGIN
                       SetColor (Col);
                       OutText ( Copy(S,1,Pos1-1) );
                       SetColor (MsgHL);
                       IF (Pos2 = 0)
                          THEN Pos2 := Length(S)+1;
                       OutText ( Copy(S,Pos1,Pos2-Pos1) );
                       SetColor (Col);
                       Delete (S, 1, Pos2-1);
                     END
           END; {While}
       END; {Ds_OutText}


     PROCEDURE Ds_WaitMessage (S : String;  VAR Ch : Char);
       CONST
         Sentence = 19;
       VAR
         S2    : String[30];
         Lines : Word;
         Count : Word;
       BEGIN
         SetTextStyle (SmallFont, HorizDir, FontSz);
         SetColor (DsCol_Fgrnd2);
         Ds_DrawWindow (MsgWind);
         ClearViewPort;
         Lines := Length(S) DIV Sentence + 2;
         IF (Lines < 6)
            THEN Lines := 6;
         Count := 1;
         SetColor (DsCol_Fgrnd2);
         SetTextJustify (LeftText, CenterText);
         WHILE (Length(S) > Sentence) DO
           BEGIN
             S2 := Copy (s, 1, Sentence);
             Delete (s, 1, Sentence);
             MoveTo ( Hinset, TRUNC (Vscale/4 * (Count/Lines) )  );
             OutText (S2);
             Inc (Count);
           END;
         MoveTo ( Hinset, TRUNC (Vscale/4 * (Count/Lines) )  );
         OutText (s);
         Ch := ReadKey;
         ClearViewPort;
         ResetViewPort;
       END; {Ds_WaitMessage}


     PROCEDURE Ds_SendMessage (S : String);
       CONST
         Sentence = 19;
       VAR
         S2    : String[30];
         Lines : Word;
         Count : Word;
       BEGIN
         SetTextStyle (SmallFont, HorizDir, FontSz);
         Ds_SetWindow (MsgWind);
         ClearViewPort;
         Lines := Length(S) DIV Sentence + 2;
         IF (Lines < 6)
            THEN Lines := 6;
         Count := 1;
         SetColor (DsCol_Fgrnd2);
         SetTextJustify (LeftText, CenterText);
         WHILE (Length(S) > Sentence) DO
           BEGIN
             S2 := Copy (s, 1, Sentence);
             Delete (s, 1, Sentence);
             MoveTo ( Hinset, TRUNC (Vscale/4 * (Count/Lines) )  );
             OutText (S2);
             Inc (Count);
           END;
         MoveTo ( Hinset, TRUNC (Vscale/4 * (Count/Lines) )  );
         OutText (s);
         ResetViewPort;
       END; {Ds_SendMessage}


     PROCEDURE Ds_AddMessage (Snum, Sqty : Word; S : String);
       BEGIN
         SetTextStyle (SmallFont, HorizDir, FontSz-1);
         SetColor (DsCol_Fgrnd2);
         SetTextJustify (LeftText, CenterText);
         Ds_SetWindow (MsgWind);
         MoveTo ( Hinset, TRUNC (MsgWind.Sz[2] * Snum/(Sqty+1) )  );
         OutText (s);
         ResetViewPort;
       END; {Ds_AddMessage}


     PROCEDURE Ds_WaitResponse (VAR Ch : Char);
       BEGIN
         Ds_SetWindow (MsgWind);
         Ch := Ds_ReadKey;
         ClearViewPort;
         ResetViewPort;
       END; {Ds_WaitResponse}

     PROCEDURE Ds_GetInput (  Msg : String;
                            VAR S : String);
       VAR
         Ch : Char;
       BEGIN
         SetTextStyle (SmallFont, HorizDir, FontSz);
         Ds_SetWindow (MsgWind); ClearViewPort;
         MoveTo (Hinset, TRUNC(MsgWind.Sz[2]*0.4) ); OutText (Msg);
         MoveTo (Hinset, TRUNC(MsgWind.Sz[2]*0.6) );
         S := ''; Ch := ReadKey;
         WHILE (Ch <> #13) DO
           BEGIN  S := S + Ch; OutText(Ch); Ch := ReadKey;  END;
         ClearViewPort;
         ResetViewPort;
       END; {Ds_GetInput}


   {Macro Routines for graphing}

     PROCEDURE Ds_CentreBar (Hz_st, Vt_st, Len, Ang: Single);

       PROCEDURE DoubleBar;
         CONST
           BarWidth = 0.5;
         VAR
           SinAng, CosAng : Single;
           Hz_Len, Vt_Len : Single;
           Cnr            : ARRAY [1..2] OF Pixels;
         BEGIN
           Len      := ABS(Len);
           CosAng   := COS(Ang);
           SinAng   := SIN(Ang);
           Hz_Len   := Len * CosAng;
           Vt_Len   := Len * SinAng;
           Cnr[1,1] := ROUND (Hz_st + Hz_Len*0.5 - (BarWidth * SinAng) );
           Cnr[1,2] := ROUND (Vt_st + Vt_Len*0.5 + (BarWidth * CosAng) );
           Cnr[2,1] := ROUND (Hz_st + Hz_Len*0.5 + (BarWidth * SinAng) );
           Cnr[2,2] := ROUND (Vt_st + Vt_Len*0.5 - (BarWidth * CosAng) );
           Line (Cnr[1,1], Cnr[1,2], ROUND (Cnr[1,1] - Hz_Len), ROUND (Cnr[1,2] - Vt_Len) );
           Line ( ROUND (Cnr[2,1] - Hz_Len), ROUND (Cnr[2,2] - Vt_Len), Cnr[2,1], Cnr[2,2]);
         END; {DoubleBar}

       PROCEDURE SingleBar;
         VAR
           Hz_Len, Vt_Len: Single;
         BEGIN
           Hz_Len := (Len * COS(Ang));
           Vt_Len := (Len * SIN(Ang));
           Hz_St := Hz_St + Hz_Len * 0.5;
           Vt_St := Vt_St + Vt_Len * 0.5;
           Line ( ROUND(Hz_St), ROUND(Vt_St), ROUND (Hz_St - Hz_Len), ROUND (Vt_St - Vt_Len) );
         END; {SingleBar}

       BEGIN {Ds_CentreBar}
         Ang := +Ang;     {Angles defined here as: +ve CCW}
         IF (Len <= 0)
            THEN SingleBar
            ELSE DoubleBar;
       END; {Ds_CentreBar}


     PROCEDURE Ds_DrawArrow (Hz_tip, Vt_tip, Hz_len, Vt_len: Single);
       CONST
         ArrowAng     = 0.4; {10 Degrees}
         CosArrow     = 0.92106099400;   {COS (ArrowAng)}
         SinArrow     = 0.38941834231;   {SIN (ArrowAng)}
         MinArrowLen  = 2;   {2  Pixels}
       VAR
         Ang, Arrowlen  : Single;
         CosAng, SinAng : Single;
       BEGIN
         ArrowLen := SQRT ( SQR(Hz_len) + SQR(Vt_len) )/6;
         Ang      := Arctan0 (Vt_len, Hz_len);
         CosAng   := COS (Ang);
         SinAng   := SIN (Ang);
         IF (ArrowLen < MinArrowLen)
            THEN ArrowLen := MinArrowLen;
         Line ( ROUND (Hz_Tip - ArrowLen * CosArrow * CosAng + ArrowLen * SinArrow * SinAng),
                ROUND (Vt_Tip - ArrowLen * CosArrow * SinAng - ArrowLen * SinArrow * CosAng),
                ROUND (Hz_Tip), ROUND (Vt_Tip) );
         Line ( ROUND (Hz_Tip), ROUND (Vt_Tip),
                ROUND (Hz_Tip - ArrowLen * CosArrow * CosAng - ArrowLen * SinArrow * SinAng),
                ROUND (Vt_Tip - ArrowLen * CosArrow * SinAng + ArrowLen * SinArrow * CosAng) );
       END; {Ds_DrawArrow}

  BEGIN

  END. {Dsprim}
